//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"iter"
	"math"
	"math/bits"
	"runtime"
	"runtime/metrics"
	"slices"
	"strings"
	"sync/atomic"
	"unsafe"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/fdlibm"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
	"golang.org/x/sys/cpu"
)

// jitSSABackend reports whether this architecture has the new pipeline.
const jitSSABackend = true

// jitEncoding is how a Value lies in memory, for generated code, and what
// of an Object it reads: an array's class, flags, length and elements,
// which no Go code changes while native code runs, since nothing else runs
// on a runtime's goroutine then and no other goroutine may use the runtime.
// TestJITObjectLayout holds the fields' widths to what native code loads.
var jitEncoding = abi.Encoding{
	ValueSize:     int32(unsafe.Sizeof(Value{})),
	NumOffset:     int32(unsafe.Offsetof(Value{}.num)),
	RefOffset:     int32(unsafe.Offsetof(Value{}.ref)),
	Undefined:     math.Float64bits(Undefined.num),
	Null:          math.Float64bits(Null.num),
	True:          math.Float64bits(True.num),
	False:         math.Float64bits(False.num),
	Uninitialized: math.Float64bits(uninitialized.num),
	CanonicalNaN:  canonicalNaN,
	UndefinedCell: uint64(uintptr(unsafe.Pointer(&jitUndefinedCell))),
	Object:        objectBits,
	// A short BigInt (shortBig).
	ShortBigTag:     shortIntBase,
	ShortBigLowBits: shortBigLowBits,
	ShortBigHigh:    uint64(uintptr(unsafe.Pointer(&shortBigHigh))),
	// One past the int64 range (shortBigMag), and one on the heap.
	ShortBigWideTag:  shortBigBase,
	ShortBigWideHigh: uint64(uintptr(unsafe.Pointer(&shortBigHighX))),
	HeapBigTag:       heapBigBits,
	BigLenMask:       bigLenMask,
	BigDigits:        int32(bigWordSize),
	UpvalueSlot:      int32(unsafe.Offsetof(upvalue{}.slot)),
	UpvalueClosed:    int32(unsafe.Offsetof(upvalue{}.closed)),
	UpvalueSize:      int32(unsafe.Sizeof(upvalue{})),
	RandomCount:      int32(unsafe.Offsetof(mathRandom{}.count)),
	RandomNumbers:    int32(unsafe.Offsetof(mathRandom{}.numbers)),
	ObjectShape:      int32(unsafe.Offsetof(Object{}.shape)),
	ObjectProps:      int32(unsafe.Offsetof(Object{}.valp)),
	ObjectPropsLen:   int32(unsafe.Offsetof(Object{}.valn)),
	ObjectPropsCap:   int32(unsafe.Offsetof(Object{}.valc)),
	ShapeProto:       int32(unsafe.Offsetof(shape{}.proto)),
	WriteBarrier:     jit.WriteBarrier(),
	CallGo:           jit.CallGo(),
	BigFast:          uint64(jitBigFastAddr()),
	LeafFast:         uint64(jitLeafFastAddr()),
	GoFast:           uint64(jitGoFastAddr()),
	StoreRef:         jit.StoreRef(),
	Round:            runtime.GOARCH == "arm64" || cpu.X86.HasSSE41,
	ShapeKeys:        int32(unsafe.Offsetof(shape{}.keys)),
	ShapeNext:        int32(unsafe.Offsetof(shape{}.next)),
	ShapeNextKey:     int32(unsafe.Offsetof(shape{}.nextKey)),
	ShapeNextFlags:   int32(unsafe.Offsetof(shape{}.nextFlags)),
	ShapeIndex:       int32(unsafe.Offsetof(shape{}.index)),
	LinearScan:       linearScanLimit,
	PropDefault:      uint8(propDefault),
	KeySize:          int32(unsafe.Sizeof(propKey{})),
	PropertyKey:      int32(unsafe.Offsetof(propKey{}.key)),
	PropertyFlags:    int32(unsafe.Offsetof(propKey{}.flags)),

	Accessor:       accessorBits,
	AccessorGetter: int32(unsafe.Offsetof(accessor{}.getter)),
	AccessorSetter: int32(unsafe.Offsetof(accessor{}.setter)),

	ClassObject:     uint8(ClassObject),
	ClassProxy:      uint8(ClassProxy),
	ClassFunction:   uint8(ClassFunction),
	PropNotData:     uint8(propAccessor | propPrivate | propDeleted),
	PropNotWritable: uint8(propAccessor | propPrivate | propDeleted | propUninit | propWritable),
	PropWritable:    uint8(propWritable),
	PropUninit:      uint8(propUninit),

	String:         math.Float64bits(mkTag(KindString, 0)),
	Symbol:         math.Float64bits(mkTag(KindSymbol, 0)),
	StringData:     int32(unsafe.Offsetof(String{}.s)),
	StringLeft:     int32(unsafe.Offsetof(String{}.left)),
	StringLength:   int32(unsafe.Offsetof(String{}.length)),
	StringASCII:    int32(unsafe.Offsetof(String{}.ascii)),
	StringU16:      int32(unsafe.Offsetof(String{}.aux)),
	StringBuilding: int32(unsafe.Offsetof(String{}.building)),

	ObjectClass:        int32(unsafe.Offsetof(Object{}.class)),
	ObjectFlags:        int32(unsafe.Offsetof(Object{}.flags)),
	ObjectExt:          int32(unsafe.Sizeof(Object{})),
	ObjectArrayLen:     int32(unsafe.Sizeof(Object{}) + unsafe.Offsetof(arrayExt{}.arrayLen)),
	ObjectElems:        int32(unsafe.Sizeof(Object{}) + unsafe.Offsetof(objectExt{}.elems)),
	ExtElems:           int32(unsafe.Offsetof(objectExt{}.elems)),
	ObjectData:         int32(unsafe.Offsetof(objectExt{}.data)),
	IterState:          int32(unsafe.Offsetof(iterObject{}.st)),
	ClassCursor:        uint8(ClassCursor),
	IterDone:           int32(unsafe.Offsetof(iterState{}.done)),
	IterArr:            int32(unsafe.Offsetof(iterState{}.arr)),
	IterIdx:            int32(unsafe.Offsetof(iterState{}.idx)),
	IterForIn:          int32(unsafe.Offsetof(iterState{}.forIn)),
	ForInState:         int32(unsafe.Offsetof(forInObject{}.in) - unsafe.Offsetof(iterObject{}.st)),
	ForInKeys:          int32(unsafe.Offsetof(forInState{}.keys)),
	ForInObj:           int32(unsafe.Offsetof(forInState{}.obj)),
	ForInShape:         int32(unsafe.Offsetof(forInState{}.shape)),
	ForInIndexKeys:     int32(unsafe.Offsetof(forInState{}.indexKeys)),
	IterIter:           int32(unsafe.Offsetof(iterState{}.iter)),
	IterNext:           int32(unsafe.Offsetof(iterState{}.next)),
	MapIterData:        jitTypeWord((*mapIterData)(nil)),
	MapIterMap:         int32(unsafe.Offsetof(mapIterData{}.m)),
	MapIterKind:        int32(unsafe.Offsetof(mapIterData{}.kind)),
	MapIterDone:        int32(unsafe.Offsetof(mapIterData{}.done)),
	MapIterAt:          int32(unsafe.Offsetof(mapIterData{}.at) + unsafe.Offsetof(mapCursor{}.i)),
	MapIterGen:         int32(unsafe.Offsetof(mapIterData{}.at) + unsafe.Offsetof(mapCursor{}.gen)),
	MapIterLast:        int32(unsafe.Offsetof(mapIterData{}.at) + unsafe.Offsetof(mapCursor{}.last)),
	MapEntries:         int32(unsafe.Offsetof(jsMap{}.entries)),
	MapGen:             int32(unsafe.Offsetof(jsMap{}.gen)),
	MapEntrySize:       int32(unsafe.Sizeof(mapEntry{})),
	MapEntryValue:      int32(unsafe.Offsetof(mapEntry{}.value)),
	MapEntrySeq:        int32(unsafe.Offsetof(mapEntry{}.seq)),
	MapKeys:            uint8(mapIterKeys),
	MapValues:          uint8(mapIterValues),
	MapDeleted:         math.Float64bits(deletedKey.num),
	GenObject:          int32(unsafe.Offsetof(generatorObject{}.g)),
	ClassGenerator:     uint8(ClassGenerator),
	GenClosure:         int32(unsafe.Offsetof(generator{}.cl)),
	GenThis:            int32(unsafe.Offsetof(generator{}.this)),
	GenVals:            int32(unsafe.Offsetof(generator{}.vals)),
	GenArgs:            int32(unsafe.Offsetof(generator{}.nargs)),
	GenPC:              int32(unsafe.Offsetof(generator{}.pc)),
	GenRare:            int32(unsafe.Offsetof(generator{}.rare)),
	GenSaved:           int32(unsafe.Offsetof(generator{}.saved)),
	SavedStack:         int32(unsafe.Offsetof(genSaved{}.stack)),
	SavedHandlers:      int32(unsafe.Offsetof(genSaved{}.handlers)),
	GenState:           int32(unsafe.Offsetof(generator{}.state)),
	GenStarted:         int32(unsafe.Offsetof(generator{}.started)),
	GenDelegating:      int32(unsafe.Offsetof(generator{}.delegating)),
	GenSuspendedStart:  uint8(genSuspendedStart),
	GenSuspendedYield:  uint8(genSuspendedYield),
	GenExecuting:       uint8(genExecuting),
	GenCompleted:       uint8(genCompleted),
	GenRoom:            int32(unsafe.Offsetof(generator{}.room)),
	GenRoomCursor:      int32(unsafe.Offsetof(generatorLoop{}.cur)),
	GenIterFrom:        int32(unsafe.Offsetof(genIterationCache{}.from)),
	GenIterCell:        int32(unsafe.Offsetof(genIterationCache{}.cell)),
	GenIterNextObj:     int32(unsafe.Offsetof(genIterationCache{}.next) + unsafe.Offsetof(ownProp{}.o)),
	GenIterNextIdx:     int32(unsafe.Offsetof(genIterationCache{}.next) + unsafe.Offsetof(ownProp{}.i)),
	GenIterIterObj:     int32(unsafe.Offsetof(genIterationCache{}.iter) + unsafe.Offsetof(ownProp{}.o)),
	GenIterIterIdx:     int32(unsafe.Offsetof(genIterationCache{}.iter) + unsafe.Offsetof(ownProp{}.i)),
	HandlerSize:        int32(unsafe.Sizeof(handler{})),
	HandlerPC:          int32(unsafe.Offsetof(handler{}.pc)),
	HandlerDepth:       int32(unsafe.Offsetof(handler{}.stackDepth)),
	HandlerFinally:     int32(unsafe.Offsetof(handler{}.isFinally)),
	FuncData:           jitTypeWord((*funcData)(nil)),
	FuncClosure:        int32(unsafe.Offsetof(funcData{}.closure)),
	ClosureFn:          int32(unsafe.Offsetof(closure{}.fn)),
	ClosureUpvalues:    int32(unsafe.Offsetof(closure{}.upvalues)),
	ClosureEnv:         int32(unsafe.Offsetof(closure{}.env)),
	ClosureRealm:       int32(unsafe.Offsetof(closure{}.realm)),
	ClosureCode:        int32(unsafe.Offsetof(closure{}.closureCode)),
	ClosureHint:        int32(unsafe.Offsetof(closureCode{}.jitHint)),
	Mega:               true,
	FuncExtra:          int32(unsafe.Offsetof(funcData{}.extra)),
	ExtraLexThis:       int32(unsafe.Offsetof(funcExtra{}.lexThis)),
	ExtraLexThisRef:    int32(unsafe.Offsetof(funcExtra{}.lexThisRef)),
	ThisBindingValue:   int32(unsafe.Offsetof(thisBinding{}.value)),
	ThisBindingInit:    int32(unsafe.Offsetof(thisBinding{}.init)),
	HintSlotSize:       int32(unsafe.Sizeof(jitHintSlot{})),
	HintSlotFn:         int32(unsafe.Offsetof(jitHintSlot{}.fn)),
	HintSlotLink:       int32(unsafe.Offsetof(jitHintSlot{}.link)),
	LinkEntry:          int32(unsafe.Offsetof(jitLink{}.entry)),
	LinkBody:           int32(unsafe.Offsetof(jitLink{}.body)),
	LinkIn:             int32(unsafe.Offsetof(jitLink{}.in)),
	LinkParams:         int32(unsafe.Offsetof(jitLink{}.params)),
	LinkLocals:         int32(unsafe.Offsetof(jitLink{}.locals)),
	LinkFrame:          int32(unsafe.Offsetof(jitLink{}.frame)),
	LinkFlags:          int32(unsafe.Offsetof(jitLink{}.flags)),
	LinkGen:            int32(unsafe.Offsetof(jitLink{}.gen)),
	ClassArray:         uint8(ClassArray),
	ClassTypedArray:    uint8(ClassTypedArray),
	TypedKind:          int32(unsafe.Offsetof(typedArrayData{}.kind)),
	TypedElems:         int32(unsafe.Offsetof(typedArrayData{}.data)),
	TypedLength:        int32(unsafe.Offsetof(typedArrayData{}.length)),
	TypedEpoch:         int32(unsafe.Offsetof(typedArrayData{}.epoch)),
	FlagSparse:         uint8(objHasSparseElements),
	FlagExt:            uint8(objHasExt),
	FlagData:           uint8(objHasData),
	FlagPoly:           uint8(objPolyProto >> 8),
	FlagHTMLDDA:        uint8(objHTMLDDA),
	FlagShared:         uint8(objSharedElems),
	FlagExtensible:     uint8(objExtensible),
	LengthKey:          uint32(atomLength),
	FlagLengthWritable: uint8(objArrayLengthWritable),
}

// jitMaxDeopts is how many eager deoptimizations a function's code makes,
// each invalidating it (jitInvalidate), before the function is compiled no
// more and runs in the existing tiers for good, as V8 gives up optimizing
// a function that keeps deoptimizing. Each compile meanwhile waits for the
// existing tiers' budget, and for its caches' feedback to have settled
// (jitTierUp): a function whose feedback has settled deoptimizes no more,
// and one meeting new shapes in phases -- a compiler's walking a syntax
// tree of new kinds of node -- deoptimizes about once a phase.
const jitMaxDeopts = 64

// jitDeoptimized notes a guard that failed in e's code, at the site
// ExitSite names, or, for -1, an entry's speculation at the entry the
// stretch started at: the code, invalidated, is compiled again with the
// site, or the entry, generic (jitFeedback.Generic), as V8's feedback
// widens once its speculation has failed. A site the builder has no
// generic form for is built as before, and its next failure is not new.
func jitDeoptimized(e *jitEntry, ctx *abi.Context, start int) {
	if site := int32(int64(ctx.ExitSite)); site >= 0 {
		if !slices.Contains(e.failed, site) {
			e.failed = append(e.failed, site)
			e.reopt = true
		}
	} else if !slices.Contains(e.failedEntries, uint32(start)) {
		e.failedEntries = append(e.failedEntries, uint32(start))
		e.reopt = true
	}
}

// jitPolySite is a read that met objects of shapes the code was not
// compiled for (jitEntry.poly): where each had the property, as a cache
// filled for it has it (fillPropCache).
type jitPolySite struct {
	pc    uint32
	cases []propCache
	// reads are the reads' cases past what a cache holds (jitDeepRead).
	reads []jitDeepRead
	// adds, for a write, are additions along chains deeper than a cache
	// holds (jitDeepAdd).
	adds []jitDeepAdd
	// accessors are the getters a read, or the setters a write, found on
	// the prototypes of receivers of shapes the code was not compiled for
	// (jitAccessorCase), which it is compiled again for.
	accessors []jitAccessorCase
	// mega marks a read that met what no case holds, which the
	// megamorphic cache has (ssa.PropertySite's Megamorphic), or a site
	// that met what the code cannot use (jitDeoptSeen); missed are the
	// receivers' shapes the site deoptimized for, each with the compile of
	// the code that did (jitEntry's ssaSerial).
	mega   bool
	missed []jitMissed
	// prim marks a read that met a receiver that is a primitive, but not
	// undefined or null, whose read throws: Go reads such a receiver's
	// property on the read's branch of its own (ssa.PropertySite's
	// Primitives), rather than at an exit.
	prim bool
}

// jitMissed is a shape a property site deoptimized for, in the code of the
// compile serial names (jitPolySite's missed).
type jitMissed struct {
	shape  *shape
	serial uint32
}

// jitMissedShapes is how many shapes a site deoptimizes for before it is
// megamorphic: one more than its cases hold.
const jitMissedShapes = jitPropertyCases + 1

// jitDeepAdd is an addition of a property along a shared transition, from
// from to next, to an object whose prototype chain is up, each with the
// shape in shapes, none intercepting it: what fillStoreCache finds, but
// for a chain of up to jitAddDepth, which a cache, holding two, cannot.
type jitDeepAdd struct {
	from, next *shape
	up         []*Object
	shapes     []*shape
}

// jitAddDepth is how deep a prototype chain an addition is compiled for.
const jitAddDepth = 6

// jitDeepAddOf is the addition of key a write just made to o, of shape
// before then, if it took a shared transition along a chain of plain
// prototypes up to jitAddDepth that none intercepts, as fillStoreCache
// requires of a cache's -- or false.
func (r *Runtime) jitDeepAddOf(o *Object, before *shape, key Atom) (jitDeepAdd, bool) {
	return r.jitDeepAddFrom(before, o.shape, o.proto(), key)
}

// jitDeepAddFrom is jitDeepAddOf's addition of key, from shape before to
// next, to an object whose prototype is proto, with the chain as it is now.
func (r *Runtime) jitDeepAddFrom(before, next *shape, proto *Object, key Atom) (jitDeepAdd, bool) {
	const plain = propAccessor | propPrivate | propDeleted | propUninit
	if before == nil || before == noShape || before.unique || next == nil || next.unique || next.parent != before || next.key != key ||
		next.flags&(plain|propWritable) != propWritable {
		return jitDeepAdd{}, false
	}
	a := jitDeepAdd{from: before, next: next}
	for p := proto; p != nil; p = p.proto() {
		if len(a.up) == jitAddDepth || r.ensureProtoShape(p) == nil || !shapeClass(p.class) || synthesized(p.class, key) || !passesWrite(p, key) {
			return jitDeepAdd{}, false
		}
		a.up, a.shapes = append(a.up, p), append(a.shapes, p.shape)
	}
	return a, true
}

// jitPolyWriteSeen notes, after Go made the write at pc that left native
// code, what its cache knows of the object it wrote -- an own property, or
// the transition it was added along -- for the code to be compiled again
// for it with the shapes the site met before (writeCases), up to
// jitPropertyCases, as for reads (jitPolySeen). DeltaBlue's Constraint
// adds strength to the receivers of four constructors, each of a shape of
// its own; RayTrace writes hitCount to infos of two.
func (r *Runtime) jitPolyWriteSeen(f *frame, e *jitEntry, pc int, in bytecode.Instr, o *Object, before *shape) {
	if int(in.B) >= len(f.cl.ic) {
		return
	}
	key, named := Atom(0), int(in.A) < len(f.cl.names)
	if named {
		key = f.cl.names[in.A]
	}
	r.jitPolyWriteSeenOp(e, pc, &f.cl.ic[in.B], key, named, o, before)
}

// jitPolyWriteSeenOp is jitPolyWriteSeen's, of either form's code: the
// write at pc, its site's cache cache, of key where named, to o, of shape
// before then.
func (r *Runtime) jitPolyWriteSeenOp(e *jitEntry, pc int, cache *propCache, key Atom, named bool, o *Object, before *shape) {
	i := slices.IndexFunc(e.poly, func(p jitPolySite) bool { return int(p.pc) == pc })
	if i < 0 {
		e.poly = append(e.poly, jitPolySite{pc: uint32(pc)})
		i = len(e.poly) - 1
	}
	p := &e.poly[i]
	if o != nil && named {
		// A setter on the object's prototypes, as for a read's getter: of
		// the shape the object had, which the setter may have changed.
		if a, ok := r.jitAccessorCaseOf(o, before, key, true); ok {
			if jitAddAccessorCase(p, a) {
				e.polyReopt = true
			}
			return
		}
	}
	if len(p.cases)+len(p.adds) >= jitPropertyCases {
		return
	}
	c := *cache
	if c.shape == nil || c.shape == noShape || c.getter || c.next == setterNext || c.idx < 0 || c.next == nil && c.p1 != nil {
		// No cache: an addition along a chain deeper than one holds, as
		// the object took it.
		if o == nil || !named {
			return
		}
		a, ok := r.jitDeepAddOf(o, before, key)
		if !ok || slices.ContainsFunc(p.adds, func(k jitDeepAdd) bool { return k.from == a.from }) {
			return
		}
		p.adds = append(p.adds, a)
		e.polyReopt = true
		return
	}
	if slices.ContainsFunc(p.cases, func(k propCache) bool { return k.shape == c.shape }) {
		return
	}
	p.cases = append(p.cases, c)
	e.polyReopt = true
}

// jitWriteTarget is the object a set_prop about to be made writes, its
// operands the two values below sp, and its shape now; nil for another
// instruction or a primitive.
func jitWriteTarget(stack []Value, sp int, in bytecode.Instr) (*Object, *shape) {
	if in.Op != bytecode.OpSetProp || sp < 2 || sp > len(stack) || !stack[sp-2].IsObject() {
		return nil, nil
	}
	o := stack[sp-2].Object()
	return o, o.shape
}

// jitPropertyCases is how many shapes a read is compiled for, as V8's
// polymorphic inline caches keep: one more leaves for Go.
const jitPropertyCases = 4

// jitPolySeen notes a read native code left to Go at pc, its receiver on top
// of the frame's operands, up to sp -- a megamorphic site's, whose others
// deoptimize (jitPropertyDeopt), or an inlined callee's: the megamorphic
// cache is filled for the receiver's shape, which the read then finds
// natively; what that cannot hold -- a getter, a read past the prototypes
// it holds, a string's method -- the code is compiled again for (polyReopt),
// on the existing tiers' budget, its code invalidated meanwhile.
func (r *Runtime) jitPolySeen(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) {
	op := jitPropOp{recv: r.stack[sp-1], method: in.Op == bytecode.OpGetPropThis, unnamed: int(in.A) >= len(f.cl.names)}
	if !op.unnamed {
		op.key = f.cl.names[in.A]
	}
	if int(in.B) < len(f.cl.ic) {
		op.c = &f.cl.ic[in.B]
	}
	r.jitPolySeenOp(e, pc, op)
}

// jitPropOp is a named property's operation as what the JIT learns at its
// exits sees it, of either form's code: its key, its receiver, its site's
// cache (nil where it has none, as an object pattern's read), and whether
// it is a write, a method's read (get_prop_this's, get_method's), or an
// object pattern's read (get_prop_under's); unnamed marks one whose key is
// past its closure's names.
type jitPropOp struct {
	key                           Atom
	recv                          Value
	c                             *propCache
	write, method, under, unnamed bool
}

// plain reports whether op is a plain read: get_prop's, not a method's.
func (op *jitPropOp) plain() bool { return !op.write && !op.method && !op.under }

// jitPolySeenOp is jitPolySeen's, of either form's code: the read op at pc
// that native code left to Go.
func (r *Runtime) jitPolySeenOp(e *jitEntry, pc int, op jitPropOp) {
	recv := op.recv
	if recv.IsString() && op.method && !slices.Contains(e.strSites, uint32(pc)) {
		// A string's method: read off String.prototype by the code
		// compiled again, as for a shape the read had not met.
		e.strSites = append(e.strSites, uint32(pc))
		e.polyReopt = true
		return
	}
	if recv.IsBigInt() && op.method && !slices.Contains(e.bigSites, uint32(pc)) {
		// A BigInt's: off BigInt.prototype, as a string's.
		e.bigSites = append(e.bigSites, uint32(pc))
		e.polyReopt = true
		return
	}
	if r.jitPrimitiveSeen(e, pc, &op) {
		// A number's or a boolean's: off Number.prototype or
		// Boolean.prototype, as a string's.
		e.polyReopt = true
		return
	}
	if !recv.IsObject() || op.unnamed {
		if !recv.IsObject() && !recv.IsNullish() && op.plain() {
			// A string's, a number's, a boolean's: Go reads it from
			// native code once the code is compiled again (prim) -- an
			// element type, React's createElement's a.defaultProps of
			// "div" -- where it had left native code at each.
			if p := jitPolySiteAt(e, pc); !p.prim {
				p.prim = true
				e.polyReopt = true
			}
		}
		return
	}
	if op.method && op.c != nil && !slices.Contains(e.strSites, uint32(pc)) {
		if c := op.c; (c.shape == nil || c.shape == noShape) && c.fills == 0 && r.proto.str != nil && r.proto.str.findOwn(op.key) >= 0 {
			// An object's method, at a read the code took for a string's
			// (jitFeedback's StringProperty), its cache knowing nothing: a
			// loop the tree tier ran before it was compiled, a function
			// run once, leaves no read in the interpreter's cache. The
			// cache takes the receiver, as the interpreter's read would
			// have, and the code is compiled again for its shape; it had
			// left there at every entry, and was judged slow for it.
			r.fillPropCache(c, recv.Object(), op.key, false)
			if c.shape != nil && c.shape != noShape {
				e.polyReopt = true
				return
			}
		} else if c.shape != nil && c.shape != noShape && c.shape == recv.Object().shape && r.proto.str != nil && r.proto.str.findOwn(op.key) >= 0 {
			// So too where the interpreter filled the cache between the
			// compile and this exit, with the receiver's shape: the code
			// still takes the read for a string's, and had left at every
			// read -- [i, 1].at(1), the interpreter running the rest of
			// each iteration -- the exit learning nothing.
			e.polyReopt = true
			return
		}
	}
	if op.plain() {
		// A getter on the receiver's prototypes, which the code calls once
		// compiled again for the receiver's shape too (jitFeedback's
		// Accessor).
		if a, ok := r.jitAccessorCaseOf(recv.Object(), recv.Object().shape, op.key, false); ok {
			if jitAddAccessorCase(jitPolySiteAt(e, pc), a) {
				e.polyReopt = true
			}
			return
		}
	}
	r.jit.megaFill(recv.Object(), op.key)
	absent := r.jitMegaFillProto(recv.Object(), op.key)
	var c propCache
	r.fillPropCache(&c, recv.Object(), op.key, false)
	if c.shape == nil {
		// One a cache cannot hold, found past its two prototypes.
		if d, ok := r.jitDeepReadOf(recv.Object(), op.key); ok && jitAddDeepRead(jitPolySiteAt(e, pc), d) {
			e.polyReopt = true
		} else if p := jitPolySiteAt(e, pc); absent && !p.mega {
			// None its cases could hold -- a property nothing has, a leaf's
			// child -- which the megamorphic cache now says: the code is
			// compiled again to look there (ssa.PropertySite's
			// Megamorphic), rather than leave at every such read.
			p.mega = true
			e.polyReopt = true
		}
		return
	}
	if c.shape == noShape || c.getter || c.idx < 0 {
		return
	}
	holder := recv.Object()
	if c.p2 != nil {
		holder = c.p2
	} else if c.p1 != nil {
		holder = c.p1
	}
	if c.idx < int32(holder.valn) && !holder.vals()[c.idx].IsNumber() {
		jitNumberMissed(e, pc)
	}
	// A shape it was not compiled for: at a megamorphic site, the
	// megamorphic cache, filled above, has it natively from now on.
}

// jitPrimitiveSeen notes in e the method read in at pc, of f's function,
// as one that met recv, a number or a boolean, and reports whether it did:
// one whose key the prototype the realm fixed for recv's kind,
// Number.prototype or Boolean.prototype, has as a plain data property,
// which the code compiled again reads off that prototype, as a string's
// off String.prototype (jitFeedback's StringProperty). A site that has met
// another primitive already is read as StringProperty takes it, the other
// kinds' receivers left to Go; a method the prototype does not have, Go
// reads.
func (r *Runtime) jitPrimitiveSeen(e *jitEntry, pc int, op *jitPropOp) bool {
	var sites *[]uint32
	var proto *Object
	switch recv := op.recv; {
	case recv.IsNumber():
		sites, proto = &e.numSites, r.proto.number
	case recv.IsBool():
		sites, proto = &e.boolSites, r.proto.boolean
	default:
		return false
	}
	if !op.method || op.unnamed || proto == nil {
		return false
	}
	at := uint32(pc)
	if slices.Contains(e.strSites, at) || slices.Contains(e.bigSites, at) || slices.Contains(e.numSites, at) || slices.Contains(e.boolSites, at) {
		return false
	}
	key := op.key
	i := proto.findOwn(key)
	if key.IsIndex() || i < 0 || proto.shape.keys[i].flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 {
		return false
	}
	*sites = append(*sites, at)
	return true
}

// jitPropertyOp reports whether op is a named property's read or write,
// whose site's cache the interpreter fills (propCache).
func jitPropertyOp(op bytecode.Op) bool {
	return op == bytecode.OpGetProp || op == bytecode.OpGetPropThis || op == bytecode.OpSetProp || op == bytecode.OpGetPropUnder
}

// jitPropertyDeopt handles a deoptimization at site of the code owner's
// entry holds, at pc of f's function, its operands up to sp: if the
// instruction there is a named property's read or write, its receiver was
// of a shape the site's cases do not have, or its site knew none
// (ssa.PropertySite), as V8's map checks fail and its code deoptimizes for
// insufficient feedback, and it reports true. The interpreter makes the
// operation from the frame Go wrote, its cache learning the shape -- the
// feedback the code is compiled again with (jitSeenFeedback), as V8's
// inline caches gather it -- and the code, which the caller invalidates
// as any eager deoptimization does, is compiled again for it once the
// existing tiers have spent their budget (jitTierUp). What a cache cannot
// hold, e notes (jitDeoptSeen). A site entry's speculation (site -1) is no
// property's.
func (r *Runtime) jitPropertyDeopt(f *frame, e, owner *jitEntry, site int32, pc, sp int) bool {
	if site < 0 || pc >= len(f.cl.fn.Code) || !jitPropertyOp(f.cl.fn.Code[pc].Op) {
		return false
	}
	serial := uint32(0)
	if owner != nil {
		serial = owner.ssaSerial
	}
	r.jitDeoptSeen(f, e, pc, sp, f.cl.fn.Code[pc], serial)
	if e != nil {
		r.jit.account(e)
	}
	return true
}

// jitSiteMegamorphic reports whether the property site at pc of cl's
// function is megamorphic as its code was compiled (jitFeedback's
// megamorphic): its cache gave up, or e, the function's entry, found what
// no case holds.
func (r *Runtime) jitSiteMegamorphic(cl *closure, e *jitEntry, pc int) bool {
	if in := cl.fn.Code[pc]; in.Op != bytecode.OpGetPropUnder && int(in.B) < len(cl.ic) && cl.ic[in.B].fills >= maxCacheFills {
		// An object pattern's read has no cache: its B is how far down its
		// source is.
		return true
	}
	if e != nil {
		for i := range e.poly {
			if p := &e.poly[i]; int(p.pc) == pc && p.mega {
				return true
			}
		}
	}
	return false
}

// jitPolyPrim reports whether e's read at pc met a primitive receiver
// whose read does not throw (jitPolySite's prim).
func jitPolyPrim(e *jitEntry, pc int) bool {
	for i := range e.poly {
		if p := &e.poly[i]; int(p.pc) == pc {
			return p.prim
		}
	}
	return false
}

// jitDeoptSeen notes in e what the property operation in at pc, which
// deoptimized, meets that its site's cache cannot hold, its operands up to
// sp, before the interpreter makes it: a string's method, read
// off String.prototype by the code compiled again; a getter, or a setter,
// on the receiver's prototypes (jitAccessorCase); a read the cache cannot
// hold for its prototypes' depth (jitDeepRead); a read's value that is no
// number, which the read then takes as a value (jitNumberMissed); and a
// receiver the cache cannot hold at all -- a primitive, a property nothing
// has, an object of no shape -- or one the site deoptimized for before, in
// code compiled before the code that did now (serial), its cache holding
// what the code cannot use (an object's own accessor, an addition Go
// makes), or of more shapes than its cases hold,
// which makes the site megamorphic (jitPolySite's mega), as V8's inline
// caches go megamorphic for what they have no handler for: the megamorphic
// cache, filled here, answers what it can natively, and the rest leaves
// for Go rather than deoptimize again. So each deoptimization at a site
// teaches its cache a shape, which its fills bound (maxCacheFills), or
// makes it megamorphic, once. The shapes the site meets, the cache keeps
// (propCache.seen); a write's cache learns them as the interpreter writes.
func (r *Runtime) jitDeoptSeen(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr, serial uint32) {
	if int(in.A) >= len(f.cl.names) {
		return
	}
	op := jitPropOp{key: f.cl.names[in.A], write: in.Op == bytecode.OpSetProp, method: in.Op == bytecode.OpGetPropThis, under: in.Op == bytecode.OpGetPropUnder}
	at := sp - 1
	if op.write {
		at = sp - 2
	}
	if op.under {
		at = sp - 1 - int(in.B)
	}
	if at < f.base || at >= len(r.stack) {
		return
	}
	op.recv = r.stack[at]
	if !op.under && int(in.B) < len(f.cl.ic) {
		op.c = &f.cl.ic[in.B]
	}
	r.jitDeoptSeenOp(e, pc, op, serial)
}

// jitDeoptSeenOp is jitDeoptSeen's, of either form's code: the operation op
// at pc, which deoptimized in the code of compile serial, before the
// interpreter makes it.
func (r *Runtime) jitDeoptSeenOp(e *jitEntry, pc int, op jitPropOp, serial uint32) {
	if e == nil {
		// Nothing to note in: the cache learns what it can.
		e = &jitEntry{jitLearned: &jitLearned{}}
	}
	key, write, recv := op.key, op.write, op.recv
	if op.under {
		// An object pattern's read, which has no cache, missed its source's
		// own table -- a property the source lacks, its default the
		// pattern's: it looks in the megamorphic cache from the next
		// compile on, filled now, and Go reads what it does not find from
		// native code (jitFeedback's Property), rather than deoptimize at
		// each such read. One that never misses keeps its own table's
		// search alone, with no branch for Go.
		jitPolySiteAt(e, pc).mega = true
		if recv.IsObject() {
			r.jit.megaFill(recv.Object(), key)
			r.jitMegaFillProto(recv.Object(), key)
		}
		return
	}
	if recv.IsBigInt() && op.method && !slices.Contains(e.bigSites, uint32(pc)) {
		// A BigInt's method: read off BigInt.prototype by the code compiled
		// again, the site no polymorphic one (jitFeedback's StringProperty).
		e.bigSites = append(e.bigSites, uint32(pc))
		return
	}
	if r.jitPrimitiveSeen(e, pc, &op) {
		// A number's or a boolean's: off Number.prototype or
		// Boolean.prototype, so.
		return
	}
	p := jitPolySiteAt(e, pc)
	if recv.IsString() && op.method && !slices.Contains(e.strSites, uint32(pc)) {
		// A string's method: read off String.prototype by the code
		// compiled again.
		e.strSites = append(e.strSites, uint32(pc))
		return
	}
	if !recv.IsObject() || op.c == nil {
		p.mega = true
		if !recv.IsObject() && !recv.IsNullish() && op.plain() {
			// A primitive's read, Go's from native code in the code
			// compiled again (jitPolySite's prim): one compile, not one to
			// be megamorphic and another for the primitive.
			p.prim = true
		}
		return
	}
	o := recv.Object()
	i := slices.IndexFunc(p.missed, func(m jitMissed) bool { return m.shape == o.shape })
	if o.shape == nil || i >= 0 && p.missed[i].serial != serial || i < 0 && len(p.missed) >= jitMissedShapes {
		// An object with no shape, which the VM gives one only once a
		// cache asks -- an empty one, made anew each time; one of a shape
		// the site deoptimized for before, compiled again since: what its
		// cache holds of it, the code cannot use; or more shapes than its
		// cases hold.
		p.mega = true
		if !write {
			r.jit.megaFill(o, key)
			r.jitMegaFillProto(o, key)
		}
		return
	}
	if i >= 0 {
		// Again in the code that deoptimized for it, not compiled again
		// yet: the cache has it, and the code will.
		return
	}
	p.missed = append(p.missed, jitMissed{remember(o.shape), serial})
	if a, ok := r.jitAccessorCaseOf(o, o.shape, key, write); ok {
		// A getter, or a setter, on its prototypes, which the code calls
		// once compiled again for the receiver's shape too (Accessor).
		jitAddAccessorCase(p, a)
		return
	}
	c := op.c
	if write || c.fills >= maxCacheFills {
		// A write's cache learns as the interpreter writes -- an own
		// property, an addition, or, for one it cannot hold, a fill all the
		// same, toward giving up (setPropCached); a cache that has given
		// up is megamorphic.
		return
	}
	// The site's cache learns the receiver's shape here, as the tree
	// tier's read would teach it (ownScan, cachedGet): the interpreter,
	// which makes the read after, reads an object's small table without
	// it (plainOwn).
	if c.holder(o) == nil {
		r.fillPropCache(c, o, key, op.plain())
	}
	if c.fills >= maxCacheFills {
		return
	}
	if c.holder(o) == nil {
		// One a cache cannot hold: past its two prototypes, or none of
		// them has it, or not of a shape the cache may hold.
		if d, ok := r.jitDeepReadOf(o, key); ok && jitAddDeepRead(p, d) {
			return
		}
		r.jit.megaFill(o, key)
		r.jitMegaFillProto(o, key)
		p.mega = true
		return
	}
	if c.getter || c.idx < 0 {
		return
	}
	holder := o
	if c.p2 != nil {
		holder = c.p2
	} else if c.p1 != nil {
		holder = c.p1
	}
	if c.idx < int32(holder.valn) && !holder.vals()[c.idx].IsNumber() {
		jitNumberMissed(e, pc)
	}
}

// megaFill has the megamorphic cache (abi.MegaCache) name o's own data
// property key, if it has one, for objects of its shape: a property
// operation that met more shapes than it was compiled for, of an object
// too large to search, finds it there natively. AST nodes of a dozen
// fields and twenty kinds, Octane's Typescript's, had their every read
// and write at such sites left to Go.
func (s *jitState) megaFill(o *Object, key Atom) {
	sh := o.shape
	if s.mega == nil || sh == nil || sh.loose || o.class != ClassObject && o.class != ClassFunction || int(o.valn) <= abi.MaxScan {
		return
	}
	i := o.findOwn(key)
	if i < 0 || int(i) >= len(sh.keys) || int(i) >= int(o.valn) {
		return
	}
	flags := sh.keys[i].flags
	if flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 {
		return
	}
	idx := uint32(i)
	if flags&propWritable != 0 && !sh.pristine {
		// Not a pristine object's, whose writes are left to Go, which
		// tells it it is no longer (shape's pristine).
		idx |= abi.MegaWritable
	}
	// Remembered, so that it never changes, as any cache's shape.
	remember(sh)
	s.megaPut(abi.MegaEntry{Shape: unsafe.Pointer(sh), Key: uint32(key), Index: idx})
}

// megaPut puts n into its set of the megamorphic cache, first, the others
// moved down: an entry of its shape and key replaced, or, with none, the
// set's last let go.
func (s *jitState) megaPut(n abi.MegaEntry) {
	set := s.mega[abi.MegaSet(uintptr(n.Shape), n.Key):][:abi.MegaWays]
	at := len(set) - 1
	for w := range set {
		if set[w].Shape == n.Shape && set[w].Key == n.Key {
			at = w
			break
		}
	}
	copy(set[1:at+1], set[:at])
	set[0] = n
}

// jitNoteChainRead has the megamorphic cache name what a read of key from o
// found up its chain, or that nothing on it has the key, as an interpreter
// cache remembers the read by the chain's cell (fillChainCache): native
// code reads such a site megamorphically (jitFeedback.megamorphic), and
// finds it there from its first run, rather than after its first exit.
func (r *Runtime) jitNoteChainRead(o *Object, key Atom) {
	if r.jit == nil {
		return
	}
	if r.jit.mega == nil {
		r.jit.mega = new(abi.MegaCache)
	}
	r.jitMegaFillProto(o, key)
}

// jitMegaFillProto has the megamorphic cache name the data property key a
// prototype of o has, or that none has, o having none of its own, for
// objects of o's shape:
// found along the chain from o's prototype on, up to abi.MegaDepth of
// them, each of a class with shapes that does not answer the key before
// it, as a cache's walk (fillPropCache, jitDeepReadOf) requires; each with
// its shape, remembered, which native code compares (mir's mega). A read
// whose site met more shapes than it was compiled for -- a method of a
// base class, read of objects of many subclasses -- then finds it
// natively, as V8's megamorphic loads do, where it had left native code.
func (r *Runtime) jitMegaFillProto(o *Object, key Atom) (absent bool) {
	s := r.jit
	sh := o.shape
	if s.mega == nil || sh == nil || sh == noShape || sh.loose || o.class != ClassObject && o.class != ClassFunction ||
		key.IsIndex() || synthesized(o.class, key) || o.findOwn(key) >= 0 {
		return false
	}
	chain := new(abi.MegaChain)
	n := 0
	for p := o.proto(); p != nil; p = p.proto() {
		if n == abi.MegaDepth || r.ensureProtoShape(p) == nil || p.cacheShape() == nil || !shapeClass(p.class) || synthesized(p.class, key) {
			return false
		}
		chain[n] = abi.MegaLink{Object: unsafe.Pointer(p), Shape: unsafe.Pointer(p.shape)}
		n++
		i := p.findOwn(key)
		if i < 0 {
			continue
		}
		if i >= p.shape.n || int(i) >= int(p.valn) || p.shape.keys[i].flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 {
			return false
		}
		remember(sh)
		for _, l := range chain[:n] {
			remember((*shape)(l.Shape))
		}
		s.megaPut(abi.MegaEntry{Shape: unsafe.Pointer(sh), Chain: chain, Key: uint32(key), Index: uint32(i)})
		return false
	}
	// On none of them: the read's value is undefined, the chain all of
	// them (abi.MegaAbsent), as V8's loads of a nonexistent property find
	// it. Box2D's tree nodes, given a child only once they have one, read
	// a leaf's child1 so at every IsLeaf, which had left native code.
	remember(sh)
	for _, l := range chain[:n] {
		remember((*shape)(l.Shape))
	}
	s.megaPut(abi.MegaEntry{Shape: unsafe.Pointer(sh), Chain: chain, Key: uint32(key), Index: abi.MegaAbsent})
	return true
}

// jitPrivateSeen notes a private member's read or write, or #x in o,
// native code left to Go at pc, its operands on the frame's up to sp: an
// object of a shape the site has not met, whose own data the member is --
// a field, or, but for a write, a method -- joins the site's cases, as a property read's do
// (jitPolySeen), and the code is compiled again for them. A key in a local,
// of a class the function itself evaluates, is Go's.
func (r *Runtime) jitPrivateSeen(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) {
	if e == nil || in.B&1 == 0 || int(in.B>>1) >= len(f.cl.upvalues) {
		return
	}
	at := sp - 1
	if in.Op == bytecode.OpSetPrivate {
		at = sp - 2
	}
	if at < 0 || !r.stack[at].IsObject() {
		return
	}
	o := r.stack[at].Object()
	sh, i := o.shape, o.findOwn(privateKey(f, f.cl, in.B))
	if sh == nil || sh == noShape || sh.loose || i < 0 || i >= sh.n || int(i) >= int(o.valn) {
		return
	}
	if p := o.shape.keys[i]; p.isAccessor() || in.Op == bytecode.OpSetPrivate && p.flags&propWritable == 0 {
		return
	} else if in.Op == bytecode.OpGetPrivate && !o.vals()[i].IsNumber() {
		// A read that met a value not a number reads it as a value.
		jitNumberMissed(e, pc)
	}
	if jitAddReadCase(jitPolySiteAt(e, pc), propCache{shape: sh, idx: i}, nil) {
		e.polyReopt = true
	}
}

// privateProperty is Property's site for a private member's read or write
// at pc: the property of the key the closure's binding holds -- a symbol
// each evaluation of the class mints, which the code checks the binding
// holds (ssa.PropertySite's Private) -- in the objects of the shapes the
// site met (jitPrivateSeen), the first its own, the rest its cases. A site
// that has met none, or whose key is in a local, is Go's.
func (fb *jitFeedback) privateProperty(pc int, in bytecode.Instr) (ssa.PropertySite, bool) {
	ref := int(in.B >> 1)
	if in.B&1 == 0 || ref >= len(fb.cl.upvalues) || fb.cl.upvalues[ref] == nil {
		return ssa.PropertySite{}, false
	}
	key := fb.cl.upvalues[ref].get()
	e := fb.e
	if fb.root != nil || e == nil {
		// An inlined callee's: what its own code met.
		e = fb.r.jit.entryFor(fb.fn)
	}
	if !key.IsSymbol() || e == nil {
		return ssa.PropertySite{}, false
	}
	var cases []propCache
	for i := range e.poly {
		if int(e.poly[i].pc) == pc {
			cases = e.poly[i].cases
		}
	}
	// A case is the shape an object had and where in it the member was,
	// met under whatever key the binding held then: every evaluation of a
	// class mints its own, and its closures share the site. One whose
	// shape holds another key at its index is no case of this key's -- an
	// object of the other evaluation's, its member read by this one's code
	// -- and is left out, so that such an object is Go's, whose brand
	// check throws.
	atom := fb.r.atoms.internSymbol(key.Symbol())
	cases = slices.DeleteFunc(slices.Clone(cases), func(c propCache) bool {
		return c.shape == nil || c.idx < 0 || c.idx >= c.shape.n || c.shape.keys[c.idx].key != atom
	})
	if len(cases) == 0 {
		return ssa.PropertySite{}, false
	}
	site := ssa.PropertySite{Key: uint32(atom), Private: uintptr(unsafe.Pointer(key.Symbol()))}
	k := fb.keep()
	for i, c := range cases {
		k.shapes = append(k.shapes, remember(c.shape))
		if i == 0 {
			site.Shape, site.Index = uintptr(unsafe.Pointer(c.shape)), c.idx
			continue
		}
		site.Cases = append(site.Cases, ssa.PropertyCase{Shape: uintptr(unsafe.Pointer(c.shape)), Index: c.idx})
	}
	return site, true
}

// jitNumberMissed notes that the read at pc met a value that is not a
// number. Lowering makes a read a number's where any use of its value
// wants one (ir.PropertyRead) -- a use on a branch never taken, too:
// RayTrace's setPixel multiplies options.pixelWidth, the string "5", only
// when it draws to a canvas -- and its code leaves for Go wherever the
// value is not, which no cache's shape changes. So the read is generic
// once it has (jitEntry.failed), as a failed guard's site is, and the code
// compiled again reads it as a value, as V8's code deoptimized for a
// value not a number reads it tagged once optimized again; its uses that
// want a number unbox it, each with its own guard.
func jitNumberMissed(e *jitEntry, pc int) {
	if slices.Contains(e.failed, int32(pc)) {
		return
	}
	if e.reg {
		// A register instruction's: its every instruction built generic.
		e.failed = append(e.failed, int32(pc))
		return
	}
	if p := e.ssaProg; e.ssa == nil || p == nil || pc >= len(p.Code) || p.Code[pc].Op != ir.PropertyRead {
		return
	}
	e.failed = append(e.failed, int32(pc))
	e.reopt = true
}

// jitPolySiteAt is e's site at pc (jitEntry.poly), made if it has none.
func jitPolySiteAt(e *jitEntry, pc int) *jitPolySite {
	i := slices.IndexFunc(e.poly, func(p jitPolySite) bool { return int(p.pc) == pc })
	if i < 0 {
		e.poly = append(e.poly, jitPolySite{pc: uint32(pc)})
		i = len(e.poly) - 1
	}
	return &e.poly[i]
}

// jitAddReadCase adds c to a read's cases (jitPolySite), up to
// jitPropertyCases, and reports whether it did. A shape another came from
// by adding properties -- an object's whose methods give it fields after
// its construction, as it moves on from its constructor's shape -- may be
// stale, no object of it left, or live, objects of it never given those
// properties; V8's inline caches keep both maps, deprecating only one whose
// field generalized. So it keeps its case while there is room, as a case
// costs only its check; where they are full, a case superseded -- by c, by
// now, the shape the site's cache holds, or by another case -- gives c its
// place, and c, superseded itself, takes none: the stale shapes had taken
// the cases, and the code left native code at every read of the live ones,
// compiled again and again for the same stale ones until its compiles ran
// out. Dropped where there was room, a shape that lived on deoptimized
// each of a function's reads of its objects in turn, one compile each --
// Typescript's Binder.bindSymbol, of symbols some of which are never given
// a type, nine times.
func jitAddReadCase(p *jitPolySite, c propCache, now *shape) bool {
	if slices.ContainsFunc(p.cases, func(k propCache) bool { return k.shape == c.shape }) {
		return false
	}
	if len(p.cases) < jitPropertyCases {
		p.cases = append(p.cases, c)
		return true
	}
	stale := func(s *shape) bool {
		return jitSuperseded(s, now) || slices.ContainsFunc(p.cases, func(k propCache) bool { return jitSuperseded(s, k.shape) })
	}
	if stale(c.shape) {
		return false
	}
	i := slices.IndexFunc(p.cases, func(k propCache) bool { return jitSuperseded(k.shape, c.shape) || stale(k.shape) })
	if i < 0 {
		return false
	}
	p.cases = append(slices.Delete(p.cases, i, i+1), c)
	return true
}

// jitSuperseded reports whether t came from s by adding properties: s is
// one of t's ancestors along the shapes' transitions. A nil t has none, and
// neither has a unique one, whose parent field is a prototype's root
// (protoRoot).
func jitSuperseded(s, t *shape) bool {
	for a := t; a != nil && !a.unique; {
		if a = a.parent; a == s {
			return a != nil
		}
	}
	return false
}

// jitCacheSeen is a cache's earlier entries (propPoly.seen), nil until
// the site has met objects of a second shape.
type jitCacheSeen = *propSeen

// seenOf is c's earlier entries (propPoly.seen), or nil.
func (c *propCache) seenOf() *propSeen {
	if c.poly == nil {
		return nil
	}
	return c.poly.seen
}

// setSeen makes s c's earlier entries, its poly made for them if it has
// none.
func (c *propCache) setSeen(s *propSeen) {
	if c.poly == nil {
		c.poly = &propPoly{}
	}
	c.poly.seen = s
}

// propSeen is a site's earlier cache entries, each of its own shape, the
// latest last.
type propSeen struct {
	n       uint8
	entries [3]propCache
	// deep are the additions the write made along prototype chains deeper
	// than a cache holds (jitNoteDeepAdd), up to jitPropertyCases; reads,
	// the properties a read found so (jitNoteDeepRead).
	deep  []jitDeepSeen
	reads []jitDeepRead
}

// jitDeepRead is a read of a plain data property an object of shape found
// on a prototype further up than a cache holds (jitDeepReadOf): at idx in
// the table of the last of up, the prototypes from the object's on, each
// of the shape in shapes when it was found, which the code checks.
type jitDeepRead struct {
	shape  *shape
	idx    int32
	up     []*Object
	shapes []*shape
}

// jitDeepReadOf is the read of key from o, if o's prototype chain answers
// it past the two prototypes a cache holds, up to jitAddDepth, with a
// plain data property, each object on the way of a class with shapes that
// does not answer the key itself, as fillPropCache requires of a cache's
// -- or false. A typed array, whose reads no cache holds, its key no
// numeric one, which names its elements, is read so at any depth: its
// methods, on its type's prototype and %TypedArray%.prototype.
func (r *Runtime) jitDeepReadOf(o *Object, key Atom) (jitDeepRead, bool) {
	typed := o.class == ClassTypedArray && r.jitTypedNamed(o, key)
	if key.IsIndex() || o.shape == nil || o.shape == noShape || !shapeClass(o.class) && !typed || synthesized(o.class, key) || o.findOwn(key) >= 0 {
		return jitDeepRead{}, false
	}
	d := jitDeepRead{shape: o.shape}
	for p := o.proto(); p != nil; p = p.proto() {
		if len(d.up) == jitAddDepth || r.ensureProtoShape(p) == nil || !shapeClass(p.class) || synthesized(p.class, key) {
			return jitDeepRead{}, false
		}
		d.up, d.shapes = append(d.up, p), append(d.shapes, p.shape)
		if i := p.findOwn(key); i >= 0 {
			if p.shape.keys[i].flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 || len(d.up) <= 2 && !typed {
				return jitDeepRead{}, false
			}
			d.idx = int32(i)
			remember(d.shape)
			for _, s := range d.shapes {
				remember(s)
			}
			return d, true
		}
	}
	return jitDeepRead{}, false
}

// jitTypedNamed reports whether key names, on o, a typed array, a
// property its prototypes answer: no canonical numeric string, which
// names an element or nothing (typedArrayIndex), nor a symbol of its own.
func (r *Runtime) jitTypedNamed(o *Object, key Atom) bool {
	if key.IsIndex() || r.typedArrayIndex(o, key).numeric {
		return false
	}
	_, ok := o.dataOf().(*typedArrayData)
	return ok
}

// jitNoteDeepRead notes, in c's entries seen, the read of key from o a
// cache could not remember, its prototype chain deeper than one holds, if
// a prototype answers it (jitDeepReadOf): a method a class two levels
// below the one that has it inherits, as DeltaBlue's StayConstraint does
// isInput and addConstraint. Code compiled for the site reads it natively
// from the first (jitFeedback.Property); before, it left native code at
// every such read, learning nothing, until its callers stopped calling it
// natively.
func (r *Runtime) jitNoteDeepRead(c *propCache, o *Object, key Atom) {
	s := c.seenOf()
	if s != nil {
		for i := range s.reads {
			if s.reads[i].shape == o.shape {
				return
			}
		}
		if len(s.reads) >= jitPropertyCases {
			return
		}
	}
	d, ok := r.jitDeepReadOf(o, key)
	if !ok {
		return
	}
	if s == nil {
		s = &propSeen{}
		c.setSeen(s)
	}
	s.reads = append(s.reads, d)
}

// jitAddDeepRead adds d to a read's deep cases (jitPolySite.reads), up to
// jitPropertyCases with its others, and reports whether it did.
func jitAddDeepRead(p *jitPolySite, d jitDeepRead) bool {
	if len(p.cases)+len(p.reads) >= jitPropertyCases || slices.ContainsFunc(p.reads, func(k jitDeepRead) bool { return k.shape == d.shape }) ||
		slices.ContainsFunc(p.cases, func(k propCache) bool { return k.shape == d.shape }) {
		return false
	}
	p.reads = append(p.reads, d)
	return true
}

// jitDeepSeen is an addition a write made, from shape from to next, to an
// object whose prototype was proto: what jitDeepAddOf finds of it, the
// chain as it is when the code is compiled (jitDeepAddFrom).
type jitDeepSeen struct {
	from, next *shape
	proto      *Object
}

// jitNoteDeepAdd notes, in c's entries seen, an addition a write made to an
// object whose prototype chain is deeper than a cache holds, which the
// cache cannot remember -- a constructor's this.x = v in a class two levels
// below another, as DeltaBlue's constraints' are -- for code compiled for
// the site to make it natively from the first (jitSeenFeedback). Before,
// native code learned such additions one exit and one compile at a time:
// a constructor's at each of its sites in turn, for each of its receivers'
// shapes.
func jitNoteDeepAdd(c *propCache, from, next *shape, proto *Object) {
	s := c.seenOf()
	if s != nil {
		for _, d := range s.deep {
			if d.from == from {
				return
			}
		}
		if len(s.deep) >= jitPropertyCases {
			return
		}
	} else {
		s = &propSeen{}
		c.setSeen(s)
	}
	s.deep = append(s.deep, jitDeepSeen{from: remember(from), next: remember(next), proto: proto})
}

// jitNoteSeen keeps old, what a cache remembered before it remembers n,
// among the entries it has seen, if it was of another shape: replacing
// the one of its shape, or the oldest when they are full.
func jitNoteSeen(p *propPoly, old, n propCache) *propPoly {
	if old.shape == nil || old.shape == noShape || old.shape == n.shape {
		return p
	}
	old.poly = nil
	if p == nil {
		p = &propPoly{}
	}
	s := p.seen
	if s == nil {
		s = &propSeen{}
		p.seen = s
	}
	for i := range s.n {
		if s.entries[i].shape == old.shape {
			s.entries[i] = old
			return p
		}
	}
	if int(s.n) < len(s.entries) {
		s.entries[s.n] = old
		s.n++
		return p
	}
	copy(s.entries[:], s.entries[1:])
	s.entries[len(s.entries)-1] = old
	return p
}

// jitSeenFeedback adds to e's sites what cl's caches have seen besides
// their entries (propCache.poly, propCache.seen): the shapes the interpreter and
// the tree tier met at each property read and write, which the code is
// compiled for from the first, as V8's optimizing compiler takes the maps
// its inline caches met, not learned one exit and one compile at a time.
// Each is taken as an exit's would be (jitPolySeen, jitPolyWriteSeen).
func (r *Runtime) jitSeenFeedback(cl *closure, e *jitEntry) {
	for pc, in := range cl.fn.Code {
		write := in.Op == bytecode.OpSetProp
		if !write && in.Op != bytecode.OpGetProp && in.Op != bytecode.OpGetPropThis || int(in.B) >= len(cl.ic) {
			continue
		}
		key, named := Atom(0), int(in.A) < len(cl.names)
		if named {
			key = cl.names[in.A]
		}
		r.jitSeenSite(e, pc, &cl.ic[in.B], write, key, named)
	}
}

// jitSeenSite is jitSeenFeedback's, of either form's code: the site at pc,
// its cache, a write's or a read's, of key where named.
func (r *Runtime) jitSeenSite(e *jitEntry, pc int, cache *propCache, write bool, key Atom, named bool) {
	if cache.poly == nil {
		return
	}
	seen, now := cache.seenOf(), cache.shape
	// What the site remembers besides its entry (poly), which every
	// tier's reads and writes fill -- the tree's scans of an object's
	// own table too, which note nothing in seen -- and what it met
	// before that (seen), in that order. A chain's cell is no entry the
	// code is compiled for (megamorphic). Taken where they are: copied
	// into a list of their own, they had been most of what a register
	// compile's asks of its inlined callees allocated.
	if seen != nil {
		for _, c := range seen.entries[:seen.n] {
			jitSeenCase(e, pc, c, write, now)
		}
	}
	if p := cache.poly; p != nil {
		for _, c := range p.entries[:p.n] {
			if c.cell == nil {
				jitSeenCase(e, pc, c, write, now)
			}
		}
	}
	if seen == nil {
		return
	}
	if !write {
		// The reads a cache could not hold, past its two prototypes.
		for _, d := range seen.reads {
			jitAddDeepRead(jitPolySiteAt(e, pc), d)
		}
		return
	}
	if len(seen.deep) == 0 || !named {
		return
	}
	// The additions along chains deeper than a cache holds, as the chain
	// is now; one that changed since is left to Go.
	i := slices.IndexFunc(e.poly, func(p jitPolySite) bool { return int(p.pc) == pc })
	if i < 0 {
		e.poly = append(e.poly, jitPolySite{pc: uint32(pc)})
		i = len(e.poly) - 1
	}
	p := &e.poly[i]
	for _, d := range seen.deep {
		if len(p.cases)+len(p.adds) >= jitPropertyCases || slices.ContainsFunc(p.adds, func(k jitDeepAdd) bool { return k.from == d.from }) {
			continue
		}
		if a, ok := r.jitDeepAddFrom(d.from, d.next, d.proto, key); ok {
			p.adds = append(p.adds, a)
		}
	}
}

// jitSeenCase is jitSeenSite's of c, a case the site at pc met, a write's
// or a read's, whose cache holds now.
func jitSeenCase(e *jitEntry, pc int, c propCache, write bool, now *shape) {
	if c.shape == nil || c.shape == noShape || c.getter || c.idx < 0 || write && (c.next == setterNext || c.next == nil && c.p1 != nil) {
		return
	}
	i := slices.IndexFunc(e.poly, func(p jitPolySite) bool { return int(p.pc) == pc })
	if i < 0 {
		e.poly = append(e.poly, jitPolySite{pc: uint32(pc)})
		i = len(e.poly) - 1
	}
	p := &e.poly[i]
	if !write {
		// What the site's objects were before they took the shape its
		// cache remembers now (now) is a case while there is room.
		jitAddReadCase(p, c, now)
		return
	}
	if len(p.cases)+len(p.adds) >= jitPropertyCases || slices.ContainsFunc(p.cases, func(k propCache) bool { return k.shape == c.shape }) {
		return
	}
	p.cases = append(p.cases, c)
}

// jitLiteral is an object literal's site, at pc, that native code makes
// the objects of from pool, each of shape root, the literal's as the
// closure compiled for had it (shapeTree.siteRoot): its fields are
// compiled for the transitions out of it.
type jitLiteral struct {
	pc   int32
	root *shape
	pool *abi.ObjectPool
	// array marks an array literal's (new_array), whose arrays have room
	// for n elements and none yet -- but for shared's, a literal of
	// constants whose arrays are made whole, sharing them
	// (sharedLiteralArray).
	array  bool
	shared bool
	n      int
	// lit is a regular expression literal's pattern (new_regexp), whose
	// objects are its RegExps.
	lit *bytecode.Constant
	// fn is the function the literal is of: e's, or a callee's e's code
	// inlines.
	fn *bytecode.Function
	// tmpl is a closure's function template (closure), whose objects are
	// its functions, made as the VM makes them in fn's frames, their
	// captured bindings' cells unwritten (jitFillClosures); env is fn's
	// closures' environment.
	tmpl Value
	env  *Object
}

// jitLiteralRoom bounds the room learned from one array result. Larger
// buffers inherit the tree tier's confirmed length (jitLiteralArrayRoom).
const jitLiteralRoom = 64

// jitLiteralAt is e's literal at pc of cl's function -- e's, or a callee's
// its code inlines -- made for cl's site if it has none, or nil if the
// literal's objects have no shape to start from.
func (r *Runtime) jitLiteralAt(e *jitEntry, cl *closure, pc int) *jitLiteral {
	for i := range e.literals {
		if int(e.literals[i].pc) == pc && e.literals[i].fn == cl.fn {
			return &e.literals[i]
		}
	}
	in := cl.fn.Code[pc]
	if in.Op == bytecode.OpNewArray && in.A <= ir.MaxArrayLiteral {
		e.literals = append(e.literals, jitLiteral{pc: int32(pc), pool: new(abi.ObjectPool), array: true, n: int(in.A), fn: cl.fn})
		l := &e.literals[len(e.literals)-1]
		if sharedLiteralSite(r, cl, pc) {
			// A literal of constants its arrays share, whole as Go makes
			// them: native code writes no element (ssa's LiteralSite's
			// Shared).
			l.shared = true
			l.pool.Source = unsafe.Pointer(&jitPoolSource{array: true, n: l.n, sharedOf: cl, sharedPC: int32(pc)})
		} else {
			l.pool.Source = unsafe.Pointer(jitLiteralArraySource[struct{}](r, cl, l.pool, pc, l.n))
		}
		r.jitFillLiteral(l)
		return l
	}
	if in.Op == bytecode.OpNewRegExp && int(in.A) < len(cl.fn.Constants) {
		// Fresh RegExps of the pattern, each with its lastIndex 0, which
		// nothing sees before the code takes one: what the literal makes
		// at each evaluation. A pattern that does not compile has Go
		// throw.
		lit := &cl.fn.Constants[in.A]
		if _, err := r.newRegExpLiteral(lit); err != nil {
			return nil
		}
		e.literals = append(e.literals, jitLiteral{pc: int32(pc), pool: new(abi.ObjectPool), lit: lit, fn: cl.fn})
		l := &e.literals[len(e.literals)-1]
		l.pool.Source = unsafe.Pointer(&jitPoolSource{lit: lit})
		r.jitFillLiteral(l)
		return l
	}
	if in.Op == bytecode.OpClosure {
		if int(in.A) >= len(cl.consts) || cl.consts[in.A].template() == nil {
			return nil
		}
		tmpl := cl.consts[in.A]
		e.literals = append(e.literals, jitLiteral{pc: int32(pc), pool: new(abi.ObjectPool), fn: cl.fn, tmpl: tmpl, env: cl.env})
		l := &e.literals[len(e.literals)-1]
		l.pool.Source = unsafe.Pointer(&jitPoolSource{tmpl: tmpl, fn: cl.fn, env: cl.env})
		r.jitFillLiteral(l)
		return l
	}
	if in.Op != bytecode.OpNewObject || int(in.B) >= len(cl.ic) {
		return nil
	}
	return r.jitObjectLiteral(e, cl, pc, in.B, int(in.A))
}

// jitObjectLiteral is the object literal at pc of cl's code, of either
// form, made in e, its site's cache cl's at site, room for n fields: native
// code makes its objects from a pool of the shape its site's objects start
// with (shapeTree.siteRoot).
func (r *Runtime) jitObjectLiteral(e *jitEntry, cl *closure, pc int, site uint32, n int) *jitLiteral {
	keySite[struct{}](r, cl, site)
	root := r.shapes.siteRoot(&cl.ic[site], r.proto.object)
	if root == nil {
		return nil
	}
	e.literals = append(e.literals, jitLiteral{pc: int32(pc), root: root, pool: new(abi.ObjectPool), n: n, fn: cl.fn})
	l := &e.literals[len(e.literals)-1]
	l.pool.Source = unsafe.Pointer(&jitPoolSource{root: root, n: l.n})
	r.jitFillLiteral(l)
	return l
}

// jitArrayLiteral is the array literal at pc of cl's code, of either form,
// of n elements, made in e: native code makes its arrays from a pool of
// arrays with room for them, none yet.
func (r *Runtime) jitArrayLiteral(e *jitEntry, cl *closure, pc, n int) *jitLiteral {
	e.literals = append(e.literals, jitLiteral{pc: int32(pc), pool: new(abi.ObjectPool), array: true, n: n, fn: cl.fn})
	l := &e.literals[len(e.literals)-1]
	l.pool.Source = unsafe.Pointer(&jitPoolSource{array: true, n: n})
	r.jitFillLiteral(l)
	return l
}

// jitRegExpLiteral is the regular expression literal at pc of cl's code,
// of either form, of pattern lit, made in e: native code takes fresh
// RegExps of the pattern, each with its lastIndex 0, from a pool -- what
// the literal makes at each evaluation. A pattern that does not compile
// has Go throw.
func (r *Runtime) jitRegExpLiteral(e *jitEntry, cl *closure, pc int, lit *bytecode.Constant) *jitLiteral {
	if _, err := r.newRegExpLiteral(lit); err != nil {
		return nil
	}
	e.literals = append(e.literals, jitLiteral{pc: int32(pc), pool: new(abi.ObjectPool), lit: lit, fn: cl.fn})
	l := &e.literals[len(e.literals)-1]
	l.pool.Source = unsafe.Pointer(&jitPoolSource{lit: lit})
	r.jitFillLiteral(l)
	return l
}

// jitFillLiteral fills l's pool with objects as the literal makes them,
// room for its n fields: Size of them, as jitFillPool fills a pool. An
// array literal's are arrays with room for its elements, none yet, of the
// layout a property cache gives an array (ensureShape), as jitFillPool's
// for new Array() are.
func (r *Runtime) jitFillLiteral(l *jitLiteral) {
	pool := l.pool
	if pool.Size == 0 {
		pool.Size = abi.PoolSize
	}
	clear(pool.Objects[:])
	if l.tmpl.template() != nil {
		r.jitFillClosures(pool, l.tmpl, l.fn, l.env)
		return
	}
	if l.lit != nil {
		for i := range int(pool.Size) {
			v, err := r.newRegExpLiteral(l.lit)
			if err != nil {
				pool.Count = 0
				return
			}
			pool.Objects[i] = unsafe.Pointer(v.Object())
		}
		pool.Count, pool.Proto = pool.Size, unsafe.Pointer(r.proto.regexp)
		return
	}
	if !l.array {
		n := l.n
		if l.root != nil {
			n = max(n, int(l.root.slack))
		}
		jitFillObjectPool(pool, r.proto.object, l.root, n)
		return
	}
	n := l.n
	src := (*jitPoolSource)(pool.Source)
	if src != nil && src.sharedOf != nil {
		// A literal of constants: its arrays as the tree tier makes them,
		// each whole (sharedLiteralArray).
		for i := range int(pool.Size) {
			var a *Object
			if src.sharedReg {
				ic := &src.sharedOf.ic[src.sharedPC]
				a = sharedShare(r, ic, ic.p1)
			} else {
				a = sharedLiteralArray(r, src.sharedOf, int(src.sharedPC))
			}
			r.ensureShape(a)
			pool.Objects[i] = unsafe.Pointer(a)
		}
		pool.Count, pool.Proto = pool.Size, unsafe.Pointer(r.proto.array)
		return
	}
	if src != nil {
		if s := src.sample; s != nil && len(s.elemsOf()) > n {
			// The last array the pool gave grew past the literal's
			// elements -- a spread's, an append's -- as the next will:
			// they are made with room for as many, as V8's allocation
			// sites size what their literals come to hold, so that native
			// code spreads into them.
			src.room = max(src.room, min(len(s.elemsOf()), jitLiteralRoom))
		}
		n = jitLiteralArrayRoom[struct{}](pool, n)
	}
	for i := range int(pool.Size) {
		a := newArrayObject(r.proto.array, n)
		a.extOf().elems = a.elemsOf()[:0]
		r.ensureShape(a)
		pool.Objects[i] = unsafe.Pointer(a)
	}
	if src != nil {
		// The pool gives its first last.
		src.sample = (*Object)(pool.Objects[0])
	}
	pool.Count, pool.Proto = pool.Size, unsafe.Pointer(r.proto.array)
}

// jitDefineSeen notes an exit at a literal's field, pc, that the code was
// compiled for before its transition was made: Go, defining it, made it,
// and the code is compiled again for it (polyReopt), as for any feedback it
// lacked.
func jitDefineSeen(e *jitEntry, pc int) {
	if i := slices.Index(e.definesUnfed, int32(pc)); i >= 0 {
		e.definesUnfed = slices.Delete(e.definesUnfed, i, i+1)
		e.polyReopt = true
	}
}

// jitRefillLiteral fills again, with more objects, the pool of the literal
// at pc that native code left at, its pool run out.
func (r *Runtime) jitRefillLiteral(f *frame, e *jitEntry, pc int) {
	for i := range e.literals {
		if l := &e.literals[i]; int(l.pc) == pc && l.fn == f.cl.fn && l.pool.Count == 0 {
			l.pool.Size = min(2*max(l.pool.Size, abi.PoolSize), abi.PoolCapacity)
			r.jitFillLiteral(l)
		}
	}
}

// TypeofCells is ssa.TypeofFeedback's: the table of the runtime's strings
// typeof answers (jitState's typeofCells), made the first time a compile
// asks for it, as typeofString makes each, which the runtime keeps.
func (fb *jitFeedback) TypeofCells() (uintptr, bool) {
	if fb.r == nil || fb.r.jit == nil {
		return 0, false
	}
	s := fb.r.jit
	if s.typeofCells == nil {
		t := new([8]Value)
		for i := range t {
			t[i] = Str(typeofOfStr[struct{}](fb.r, i))
		}
		s.typeofCells = t
	}
	return uintptr(unsafe.Pointer(s.typeofCells)), true
}

// jitFillClosures fills pool with Size functions of the template tmpl as
// the VM makes them in a frame of fn's, its closures' environment env:
// their captured bindings' cells nil, and an arrow's this, new.target and
// home object those of no frame, which native code writes as it takes one
// (mir's closureTake) -- the cells and the this; an arrow that reads the
// others its pool is not made for (jitFeedback's Closure).
func (r *Runtime) jitFillClosures(pool *abi.ObjectPool, tmpl Value, fn *bytecode.Function, env *Object) {
	f := &frame{cl: &closure{fn: fn, env: env, upvalues: make([]*upvalue, len(fn.Upvalues))}, this: Undefined, newTarget: Undefined}
	cells := make([]*upvalue, jitCaptured(fn))
	for i := range int(pool.Size) {
		pool.Objects[i] = unsafe.Pointer(r.newClosureIn(f, tmpl, cells))
	}
	pool.Count, pool.Proto = pool.Size, nil
}

// reoptPending reports whether e's code is to be compiled again, for any
// of the reasons jitReoptimize takes: what it learned at its exits to make
// natively -- a speculation that failed, a shape, a call to inline, a
// callee that inlines more -- or that it leaves too often called natively.
// Code whose guard failed is invalidated (jitInvalidate) and compiled
// again on the existing tiers' budget (jitTierUp); code that learned
// runs on, and is compiled again once it has spent a budget itself
// (jitLearnedDue).
func (e *jitEntry) reoptPending() bool {
	return e.reopt || e.inlineReopt || e.polyReopt || e.upgradeReopt || e.unwindReopt || e.resumeReopt
}

// native reports whether Go may enter e's code, and go on in it after an
// exit: code that does not pay (entrySlow) the existing tiers run, as they
// run code invalidated (jitInvalidate).
func (e *jitEntry) native() bool { return !e.entrySlow && !e.invalid }



// jitLearnedDue reports whether e's code, cl's function's, valid, is due
// to be compiled again for what it learned at its exits (reoptPending):
// once it has spent a budget (jitTierBudget) since it began to learn, with
// nothing learned meanwhile (jitLearnedMark), as invalidated code waits
// for one (jitTierUp). It is not invalidated for it, as V8 never
// invalidates code for feedback: what it learns at its exits -- a call's
// target at a call of Go, an element's kind at a host exit -- only it
// learns, as V8's baseline code gathers feedback, and the existing tiers,
// whose caches note no such thing, would not learn it in its place. So it
// runs on until its budget is spent, and is replaced by code compiled for
// all it learned. Invalidated for it, a function learned a call a
// compile, each after a budget of the existing tiers: AWFY's CD's recurse
// ran out of compiles with two of its calls still made by Go, judged
// slow, its recursion in the tree tier.
func (r *Runtime) jitLearnedDue(cl *closure, e *jitEntry) bool {
	if e.invalid || e.ssa == nil || e.pending != nil {
		return false
	}
	if !e.reoptPending() {
		e.learning = false
		return false
	}
	spent := jitLearnedSpent(cl.fn, e)
	if !e.learning {
		e.learning, e.tierBase, e.tierMark = true, spent, jitLearnedMark(cl, e)
		e.tierBudget = r.jitTierBudget(cl.fn, e)
		return false
	}
	if spent-e.tierBase < e.tierBudget {
		return false
	}
	if m := jitLearnedMark(cl, e); m != e.tierMark {
		// Learned more meanwhile: what is left raised, as at a change of
		// the caches' feedback (jitICBudget), for it all.
		e.tierBudget = max(e.tierBudget-(spent-e.tierBase), r.jitICBudget(cl.fn))
		e.tierBase, e.tierMark = spent, m
		return false
	}
	return true
}

// jitCompileDueNow reports whether a call of Go from e's code, cl's
// function's, in the run whose context is root, is to leave the code for
// it to be compiled again: due for what it learned (jitLearnedDue), and
// compiled again once the run has left -- not refused for a native call of
// it in an outer run, below root, which goes on in it later and which the
// run's leaving does not end (otherwiseInUse). It would leave at every such
// call, the rest of each invocation the interpreter's, for a compile made
// only once those calls return: a recursive closure's, EarleyBoyer's
// loop3, called natively from deeper in its own recursion, had left at
// each call of Go some 3,000 times a run. The run's own levels, which its
// leaving unwinds, do not keep it.
func (r *Runtime) jitCompileDueNow(cl *closure, e *jitEntry, root int) bool {
	if !r.jitLearnedDue(cl, e) {
		return false
	}
	s := r.jit
	if e.link == nil {
		return true
	}
	fn := e.link.key.Value()
	for i := 0; i < root && i < len(s.ssaCtxs); i++ {
		if c := s.ssaCtxs[i]; c.Live == abi.LiveCall && c.Closure != nil && (*closure)(c.Closure).fn == fn {
			return false
		}
	}
	return true
}

// jitLearnedSpent is the work e's code has done since it was compiled, as
// the existing tiers' budget counts it (jitTierBudget): its invocations,
// each fn's length -- Go's entries, native callers' calls, calls of it the
// existing tiers ran for code Go does not enter -- the work its native
// stretches did, their loops' included (jitSSAProfit, jitCalleeProfit),
// and the existing tiers' in its loops before Go entered it at one
// (osrWork). A function entered only by on-stack replacement, each
// invocation leaving its code at once for the interpreter to finish --
// microbench_extra's PrivateCounter's add, at a private read of a shape
// learned at that exit -- had spent nothing, and was never compiled
// again for it, as V8's lower tiers spend the budget however a function
// runs. So too an invocation the interpreter went on with after the code
// left it (goOns), native code entered again at a loop's header by a check
// it forced, which spends nothing of the loop's (jitTierWork): Octane
// Typescript's Symbol.prototype.visible, inlining a method a subclass
// overrides, left at its call in each such run, and in thousands of them
// was never compiled again for the call it had learned.
//
// It is not monotonic: the native calls in (jitLink's in) it counts are
// zeroed by jitUnwound and jitRetryNative, as the counts they are judged
// against are. A compile for what the code learned, pending then, its
// base (tierBase) taken before, would find its spend below that base and
// wait as many calls' worth more. Counted across seven workloads -- a
// React renderer's 160 renders, Octane's Typescript, Box2D and PdfJS, the
// V8 suite's RayTrace, DeltaBlue and Richards -- it never was: jitUnwound
// zeroes them as it asks a compile itself, which begins a budget afresh.
func jitLearnedSpent(fn *bytecode.Function, e *jitEntry) int64 {
	st := &e.ssaStats
	calls := e.starts + e.goOns + e.link.in + e.slowCalls
	return int64(calls)*int64(max(len(fn.Code), 1)) + int64(st.work+st.calleeWork) + e.osrWork
}

// jitLearnedMark is jitFeedbackMark of cl's caches with what e's code has
// learned at its exits counted in: a change of either delays the compile
// for it (jitLearnedDue), as V8's NotifyICChanged does.
func jitLearnedMark(cl *closure, e *jitEntry) uint64 {
	return jitFeedbackMark(cl) + uint64(jitLearnedCount(e))*0x9e3779b97f4a7c15
}

// jitLearnedCount is how much e's exits have learned, which grows as they
// learn more.
func jitLearnedCount(e *jitEntry) int {
	return len(e.failed) + len(e.failedEntries) + len(e.resumed) + len(e.bigints) + len(e.bigwide) + len(e.poly) + len(e.typedReads) + len(e.stringReads) +
		len(e.keyedReads) + len(e.keyedWrites) + len(e.strSites) + len(e.numSites) + len(e.boolSites) + len(e.nativeCalls) + len(e.inlines) + len(e.notInline) +
		len(e.literals) + len(e.definesUnfed)
}

// jitInlinedLearned is, for each closure the code inlines, what its
// function's entry had learned when the code was compiled
// (jitLearnedCount), appended to to.
func (s *jitState) jitInlinedLearned(inlined []*closure, to []int32) []int32 {
	for _, cl := range inlined {
		n := int32(-1)
		if e := s.entryFor(cl.fn); e != nil {
			n = int32(jitLearnedCount(e))
		}
		to = append(to, n)
	}
	return to
}

// jitInvalidate invalidates e's code, cl's function's, for an eager
// deoptimization: Go enters it no more (native), nor do native callers
// call it, its link cleared, but Go calls the function, as V8's
// deoptimizer invalidates a
// function's current code -- "only invalidated code affects tiering
// decisions" -- its tiering requests reset and its interrupt budget with
// them. The existing tiers run the function until they have spent a fresh
// budget (jitTierUp). Runs of the code under way go on in it until they
// next leave it, as V8's frames of invalidated code deoptimize lazily, at
// their next return to it.
func (r *Runtime) jitInvalidate(cl *closure, e *jitEntry) {
	if e == nil || e.ssa == nil {
		return
	}
	if e.deopts < ^uint16(0) {
		e.deopts++
	}
	if e.invalid {
		return
	}
	e.invalid = true
	if e.owned() {
		e.link.clearCode()
	}
	e.tierBudget, e.tierMark = r.jitTierBudget(cl.fn, e), jitFeedbackMark(cl)
	r.jit.invalidated++
	// What it learned for the compile to come, counted as it grows, as a
	// compile counts it: the code waits for the budget meanwhile.
	r.jit.account(e)
	r.jit.settle()
}

// jitTierBudget is the work the existing tiers do with fn, e's function,
// before it is compiled, or compiled again: jitCallThreshold invocations
// (jitInvocationsForMaglev), each worth its instructions -- as V8's
// interrupt budget is so many invocations of its bytecode's length.
func (r *Runtime) jitTierBudget(fn *bytecode.Function, e *jitEntry) int64 {
	return int64(max(r.jitCallThreshold, 1)) * int64(max(len(fn.Code), 1))
}

// jitICBudget is what a change of fn's caches' feedback raises what is
// left of its budget to (jitICThreshold invocations' worth), as
// TieringManager::NotifyICChanged raises V8's.
func (r *Runtime) jitICBudget(fn *bytecode.Function) int64 {
	return int64(max(r.jitICThreshold, 1)) * int64(max(len(fn.Code), 1))
}


// jitFeedbackMark sums what cl's function's caches hold: each entry a
// cache holds or held before (propCache.seen), by its shape, prototypes and
// index, whether it gave up (maxCacheFills), and the reads and additions
// it noted past what one holds -- its reads and writes' sites, its
// globals', and through them its calls' targets. What a cache learns
// changes it; a cache filled again with what it held before -- a fresh
// array's method, read again at each invocation -- does not, nor the
// order its entries came in.
func jitFeedbackMark(cl *closure) uint64 {
	var m uint64
	for i := range cl.ic {
		c := &cl.ic[i]
		if c.fills >= maxCacheFills {
			m += 0x9e3779b97f4a7c15 * uint64(i+1)
		}
		var held [1 + maxPolyEntries + len(propSeen{}.entries)]*shape
		n := 0
		entry := func(x *propCache) {
			if x.shape == nil || x.shape == noShape {
				return
			}
			for _, h := range held[:n] {
				if h == x.shape {
					return
				}
			}
			if n < len(held) {
				held[n] = x.shape
				n++
			}
			h := uint64(uintptr(unsafe.Pointer(x.shape)))*31 + uint64(uintptr(unsafe.Pointer(x.p1)))*37 +
				uint64(uintptr(unsafe.Pointer(x.s1)))*41 + uint64(uintptr(unsafe.Pointer(x.p2)))*43 +
				uint64(uintptr(unsafe.Pointer(x.s2)))*47 + uint64(uintptr(unsafe.Pointer(x.next)))*53 + uint64(uint32(x.idx))*59
			if x.getter {
				h++
			}
			m += h * uint64(i+1)
		}
		entry(c)
		if p := c.poly; p != nil {
			for k := range p.entries[:p.n] {
				entry(&p.entries[k])
			}
		}
		if seen := c.seenOf(); seen != nil {
			for k := range seen.entries[:seen.n] {
				entry(&seen.entries[k])
			}
			m += uint64(len(seen.reads))<<32 + uint64(len(seen.deep))<<40
		}
	}
	return m
}




// jitFrameProgram refuses a program compileSSA does not compile.
func jitFrameProgram(fn *bytecode.Function, p *ir.Program) error {
	this := 0
	if p.This {
		this = 1
	}
	if len(p.Globals) != 0 || p.Locals != fn.LocalCount+len(fn.Upvalues)+this {
		return fmt.Errorf("%w: a program of globals, or of another frame", jitRefused)
	}
	return nil
}



// errJITStaleLinks is jitInstall's refusal of a compile whose links are
// stale (staleLinks), which compileSSAFinish builds again.
var errJITStaleLinks = errors.New("jit: a link the compile calls through was let go")

// staleLinks reports whether a link fb's code calls through (its links,
// and its keep's) is not its function's any more: one the runtime let go
// (unlink) while the code was compiled in the background.
func (s *jitState) staleLinks(fb *jitFeedback) bool {
	stale := func(links []*jitLink) bool {
		for _, l := range links {
			if s.links[l.key] != l {
				return true
			}
		}
		return false
	}
	if fb == nil {
		return false
	}
	return stale(fb.links) || fb.keep() != fb && stale(fb.keep().links)
}

// jitFeedbackHook, where a test sets it, is the feedback a build asks in
// place of fb's: a proxy of it, which notes what the build asks.
var jitFeedbackHook func(fb *jitFeedback, p *ir.Program) ssa.Feedback

// jitKeepLimit, where a test sets it, limits the keep cells of every
// compile's build (ssa.KeepFeedback), as a compile again past the
// context's (abi.MaxKeeps) limits them: what is made natively past them
// is left to Go.
var jitKeepLimit int


// jitJob is a compile in its three steps, as V8's concurrent compiles
// are: prepared on the runtime's goroutine (jitPrepare), which builds the
// function from what the runtime knows; worked (work), the optimization
// and the backend, which read and write the job and nothing of the
// runtime's; installed on the runtime's goroutine again (jitInstall),
// which takes what the work found into the feedback and places the code.
type jitJob struct {
	fn *bytecode.Function
	cl *closure
	// realm is the realm the compile is for, the current one where it was
	// handed over, which its install runs in (jitDrain).
	realm     *Realm
	p         *ir.Program
	e         *jitEntry
	fb        *jitFeedback
	keepLimit int
	// w is the compile's workspace, the process's, given back once its
	// code is placed (nil then), and f the function built in it.
	w *jit.Workspace
	f *ssa.Func
	// enc and opts are the encoding and the backend's switches the compile
	// is made with, and timing and optimized the timing (mir.Timing) and the
	// hook (jitSSAOptimized) it reports to, as they were when it was
	// prepared: a worker reads no variable a test or a host may set
	// meanwhile.
	enc       abi.Encoding
	opts      mir.Options
	timing    *mir.Times
	optimized func(*bytecode.Function, *ssa.Func)
	// snap is the compile's feedback asked ahead, which the worker builds
	// from (work), or nil for a compile built as it was prepared; buildErr
	// is that build's refusal, and missed marks a build that asked what
	// snap was not asked (jitSnapMiss), built on the runtime's goroutine
	// at its install instead.
	snap     *jitSnapshot
	buildErr error
	missed   bool
	// mc is the backend's code, or err its refusal; panicked marks a
	// refusal a worker made of a panic (jitJob.run), and allocErr is the
	// panic, where it was a failed check of the register allocation's.
	mc       *mir.Code
	err      error
	panicked bool
	allocErr *mir.AllocationError
	// held marks a compile held for its callees' (jitHold), which is no
	// job's yet and will be another's: holdRound is how many times it has
	// been, nativeCallee marks a compile for native callers (jitNativeCallee),
	// and frameAt is what a frame waits at for it (jitFrameAt).
	held, nativeCallee bool
	holdRound          int
	frameAt            int
	// What a compile in the background (jitSubmit) needs besides: its
	// state, which the worker and the runtime's goroutine hand it over by,
	// and done, closed once the worker is finished with it; how many
	// checks it has waited (jitDrain); what installs it on the runtime's
	// goroutine, reporting false to be asked again at a later check
	// (jitCompiled, jitReoptimized); and, for a compile again, the
	// entry's deoptimizations when it was built and whether it has
	// decided calls (jitReoptimize's learned).
	state   atomic.Int32
	done    chan struct{}
	checks  int
	install func(*Runtime, *jitJob) bool
	deopts  uint16
	learned bool
}

// jitSnapshots has compiles in the background built on their workers, from
// snapshots of their feedback (jitSnapshot); without it, on the runtime's
// goroutine, as they are prepared. A switch for comparing the two.
var jitSnapshots = true

// jitInstall is a compile's last step: what the work found taken into the
// feedback, and the code placed, if the JIT has the room for all its entry
// is to hold then (roomFor), its code and what describes it included:
// jit.ErrCodeBudget otherwise. It gives the job's workspace back.
func (r *Runtime) jitInstall(j *jitJob) (*jit.SSACode, *jitFeedback, error) {
	defer func() {
		jit.PutWorkspace(j.w)
		j.w = nil
	}()
	fn, p, e, fb, f, keepLimit := j.fn, j.p, j.e, j.fb, j.f, j.keepLimit
	s := r.jit
	if j.allocErr != nil {
		// A failed check of the allocation's (mir.CheckAllocations, which
		// tests set) ends the run as it does a compile on the spot.
		panic(j.allocErr)
	}
	if j.panicked {
		// A worker's refusal of a panic of the compiler's, whose function
		// is as the panic left it: a bug, which the tests look for.
		s.backendPanics++
		return nil, nil, fmt.Errorf("%w: %v", jitRefused, j.err)
	}
	if j.buildErr != nil {
		// The worker's build refused it, as jitPrepare's would have.
		return nil, nil, fmt.Errorf("%w: %v", jitRefused, j.buildErr)
	}
	if j.snap != nil {
		// What the questions the worker's build asked appended to the
		// feedback's lists, as the build from the feedback would have.
		j.snap.replay()
	}
	// The pools' roots and prototypes escape analysis took, which the
	// objects it does not make are made with at an exit.
	fb.keepVirtuals(f.VirtualAsked)
	fb.virtuals, fb.called = f.Virtuals, f.Called
	// The calls it inlines, of those whose inlining it asked (Inline).
	fb.inlinedN = len(f.Inlined)
	fb.inlinedAt = fb.inlinedAt[:0]
	for _, pc := range f.InlinedAt {
		fb.inlinedAt = append(fb.inlinedAt, int32(pc))
	}
	if f.HomesEnd != 0 {
		// The homes of the objects it does not make, past its operands,
		// which its frame takes, as it does its loops' slots: its callees'
		// frames start past them.
		fb.homes = f.HomesEnd - p.Locals
		fb.loopSlots = max(fb.loopSlots, fb.homes)
	}
	mc, err := j.mc, j.err
	if err != nil && strings.Contains(err.Error(), "runtime error") {
		// A refusal the backend's own panic made: a bug, which the tests
		// look for (backendPanics).
		s.backendPanics++
	}
	if errors.Is(err, mir.ErrKeeps) && keepLimit == 0 {
		return nil, nil, err
	}
	if err == nil && mc.Spills > abi.SpillSlots {
		// The stack form, which the register form replaces, keeps the
		// bound the context's own slots set (abi.Context's SpillMore): its
		// budget, which a function past it would hold past -- a script's
		// top level of hundreds of callees, running while its entry grows
		// -- is not made for one.
		err = fmt.Errorf("%w: more than %d spill slots", mir.ErrUnsupported, abi.SpillSlots)
	}
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %v", jitRefused, err)
	}
	if s.staleLinks(fb) {
		// Built in the background through a link let go meanwhile: built
		// again (compileSSAFinish) before room is made for it and it is
		// placed, to be thrown away.
		return nil, fb, errJITStaleLinks
	}
	// The cells the backend gave the kept values (mir's keep cells).
	fb.keeps = f.Keeps
	fb.hostsLeft, fb.unresumed, fb.unentered, fb.kept = nil, nil, nil, nil
	for _, pc := range f.HostsLeft {
		fb.hostsLeft = append(fb.hostsLeft, int32(pc))
	}
	for _, pc := range f.Unresumed {
		fb.unresumed = append(fb.unresumed, int32(pc))
	}
	for _, pc := range f.Unentered {
		fb.unentered = append(fb.unentered, int32(pc))
	}
	for _, pc := range f.EntryKept {
		fb.kept = append(fb.kept, int32(pc))
	}
	s.ssaKeeps = max(s.ssaKeeps, f.Keeps)
	need := jitEntryFixed + jit.CodeFootprint(mc) + fb.compiledBytes()
	if e != nil {
		need += e.learnedBytes()
	}
	// The hint table's slot, which remember takes once it is placed: a
	// step when the table grows, not a function's bytes (learn).
	hint := s.hintRoom(e)
	need += hint
	// The contexts' room for its spill slots, which they keep (spillsFor).
	spill := s.spillRoom(mc)
	need += spill
	// The lists of native callers its callees keep, which it joins once
	// placed (setLinks).
	need += s.callersGrowth(e, fb.links)
	if !s.roomFor(e, fb.links, need) {
		// Compiled for nothing: the room it was short of is what a compile
		// of it again asks for first (jitReoptimize).
		s.discarded++
		if e != nil {
			e.reoptRoom = max(need-e.bytes, 1)
		}
		return nil, nil, jit.ErrCodeBudget
	}
	if e != nil {
		e.reoptRoom = 0
	}
	code, err := jit.NewSSACode(r.jit.arena, mc)
	if err != nil {
		return nil, nil, err
	}
	s.spillsFor(mc)
	s.inlined += uint64(len(f.Inlined))
	s.inlinedCode += uint64(f.InlinedCode)
	if JITCodeHook != nil {
		JITCodeHook(jitFunctionOf(fn), code.Text(), mc)
	}
	// What the entry holds, its function's program and the lowering that
	// holds it (jitLowering, its own bytes too), and its link, which the
	// entry makes before it is compiled (linkFor): the room a compile asks
	// for first, which had come short of all three at a full budget, and
	// then of the lowering's own bytes -- a compile that met a budget
	// within them was made and thrown away.
	s.learn(len(fn.Code), need-hint-spill+p.Footprint()+jitLoweringFixed+jitLinkFixed)
	return code, fb, nil
}

// jitBuiltHook, where a test sets it, is told each build on the runtime's
// goroutine as it is made: the snapshot's oracle, which compares it with
// the build from the snapshot its jitFeedbackHook asked.
var jitBuiltHook func(fb *jitFeedback, p *ir.Program, f *ssa.Func, err error)

// jitBuildFrom is jitBuildSSA's build of p, asking feedback, fb's or its
// snapshot (jitSnapshot), and timed in timing, if not nil: a function of the
// compile's alone, which a worker may run where feedback is a snapshot.
func jitBuildFrom(w *jit.Workspace, fn *bytecode.Function, p *ir.Program, fb *jitFeedback, feedback ssa.Feedback, timing *mir.Times) (*ssa.Func, error) {
	f, err := ssa.BuildIn(&w.SSA, p, feedback)
	if err != nil {
		return nil, err
	}
	f.FrameAt = fb.frameAt
	if f.Loops {
		fb.loopSlots = f.LoopsEnd - p.Locals
	}
	fb.callbacks, fb.unmade = f.Callbacks, f.Unmade
	f.FrameLocals = fn.LocalCount
	f.Params, f.StartOnly = fn.ParamCount, jitcompile.StartOnly(fn)
	f.Cells = p.Cells
	// Native callers call a function's code at its call entry, which makes
	// its frame (mir's callEntry): code a call makes no frame of its own
	// for -- top-level code's, a generator's, an async function's -- has
	// none, nor code whose parameter list has a prologue (ir's ParamsDead)
	// whose states have no room for the call's arguments (ir.Program's
	// PrologueFrameOnly), which the interpreter goes on with after an exit
	// there, which only a frame of the VM's has then -- where they have, a
	// frame Go makes of a native callee's is given them (jitPrologueArgs) --
	// nor code that pushes its own function object (push_callee), which a
	// native callee's context does not hold. Go makes their calls. Code
	// that reads its arguments past its parameters -- a rest parameter's,
	// an arguments object's, a super() of them -- or its function object
	// -- a super reference's, push_callee's -- has one: a native call of
	// it writes them to its context (ssa.CallSite's FrameArgs, Function).
	f.CallEntry = !fn.TopLevel && !fn.IsModule && !fn.Async && !fn.Generator && !isGeneratorTemplate(fn) && !p.PrologueFrameOnly
	if jitGenCallable(fn) && !p.Cells {
		// A generator a loop's step may resume natively: where it resumes
		// it (mir's genEntry).
		f.GenResumes = jitGenResumes(fn, p, f)
		f.GenEntry = len(f.GenResumes) != 0
	}
	if p.This {
		f.ThisSlot = fn.LocalCount + len(fn.Upvalues)
	}
	if t := timing; t != nil {
		t.Lap(mir.PhaseBuild)
	}
	return f, nil
}

// JITCompileRepeat, if not 0, has every compile of the SSA backend that
// succeeds run that many times more, its build, optimization and backend,
// what each makes thrown away: for timing the compiler (mir.Timing) over
// the compiles a program makes, each as the program makes it, with
// clocks that read too coarsely for one compile. Their time and the
// collector's cycles they run are left out of the JIT cache's clocks
// (jitRepeatCycles), so that the program's code ages as without them. Set
// it before any runtime compiles; for one runtime at a time.
var JITCompileRepeat int

// jitGCCycles is how many cycles the collector has completed.
func jitGCCycles() uint64 {
	s := []metrics.Sample{{Name: "/gc/cycles/total:gc-cycles"}}
	metrics.Read(s)
	return s[0].Value.Uint64()
}

// jitAllocated is what the heap has handed out so far, in bytes and
// objects, for the compiles JITCompileRepeat repeats.
func jitAllocated() (bytes, objects uint64) {
	s := []metrics.Sample{{Name: "/gc/heap/allocs:bytes"}, {Name: "/gc/heap/allocs:objects"}}
	metrics.Read(s)
	return s[0].Value.Uint64(), s[1].Value.Uint64()
}


// jitRefused is compileSSA's refusal of a program it does not compile:
// for good, unlike jit.ErrCodeBudget's, which the cache's releasing code
// may lift, or jit.ErrUnavailable's, which no compile will.
var jitRefused = errors.New("jit: the program is refused")

// jitFeedback is ssa.Feedback from a closure: its names' atoms, and its
// property caches. A site that has met objects of one shape, the property
// their own plain data -- and writable, for a write -- knows where it is in
// any object of that shape: the shape settles the table's layout and the
// attributes. A shape a cache remembers is marked seen, so the VM replaces
// it rather than change it when an object's layout changes; the code holds
// it by address, and the entry keeps it alive. Objects of other shapes, and
// of none -- the VM gives a small object a shape only when a cache asks --
// are searched for the key.
type jitFeedback struct {
	// targets are the calls' targets the compile was given (jitTargetsOf),
	// which its entry's code is compiled for (jitEntry's ssaTargets).
	targets []jitTarget
	// lexThis says the code makes an arrow from a pool (Closure), whose
	// this native code writes from the context's This; loops that it makes
	// a loop of a forEach (ForEach), and loopSlots how far past the frame's
	// operands' first their slots go.
	lexThis, loops bool
	loopSlots      int
	r              *Runtime
	fn             *bytecode.Function
	cl             *closure
	shapes         []*shape
	// holders are the prototypes the code compares receivers' with
	// (ssa.Holder), held by address, which the entry keeps alive.
	holders []*Object
	// definesUnfed, the root's, are the literals' fields it compiled before
	// their transitions were made; cells the string constants' it reads.
	definesUnfed []int32
	cells        []*Value
	// inlinedAt, the root's, are the calls of its own it inlines, and
	// hostsLeft its host operations it leaves to Go (ssa.Func's HostsLeft),
	// and unresumed the PCs after its calls it has no entry after (ssa.Func's
	// Unresumed).
	inlinedAt, hostsLeft, unresumed []int32
	// unentered are the loops' headers whose entries escape analysis
	// dropped (ssa.Func's Unentered); kept those it kept for the frame
	// waiting at frameAt (ssa.Func's FrameAt, EntryKept).
	unentered, kept []int32
	frameAt         int
	// root is the function's feedback for an inlined callee's, which keeps
	// what the callee's sites hold for the code; nil for the function's
	// own. strings marks an inlined callee that calls charCodeAt. links
	// are those of the functions the code calls natively (jitLink).
	root    *jitFeedback
	strings bool
	links   []*jitLink
	// inlined are the closures of the callees the code inlines, or that
	// its compile asked to inline, inlinedN how many it inlines (ssa.Func's
	// Inlined); pools the object pools its constructions take from.
	inlined  []*closure
	inlinedN int
	// regSeen, the root's, are the register callees the compile asked to
	// inline whose sites' caches it has taken (jitRegSeenInlined), and
	// regCallees their feedback (jitRegCalleeOf).
	regSeen    []*closure
	regCallees []*jitRegFeedback
	pools      []*abi.ObjectPool
	// made are the templates of the closures the code may not make, the
	// loops' callbacks inlined where their callers make them (ForEachSite's
	// Made), which an exit's frame names in their places (jitMaterialize);
	// callbacks and unmade how many the code inlines, and does not make
	// (ssa.Func's Callbacks and Unmade).
	made              []*closure
	callbacks, unmade int
	// homes is how far past the frame's operands' first the homes of the
	// objects the code does not make end (ssa.Func's HomesEnd), or 0, and
	// virtuals how many objects it does not make (ssa.Func's Virtuals).
	homes, virtuals int
	// called is how many closures the code makes that it only calls and
	// does not make (ssa.Func's Called).
	called int
	// wanted are callees of calls to inline that the compile made calls
	// instead -- past its budget (ssa's maxInlineTotal) -- which had no code
	// for native callers: compiled after it (jitWantedCallees). linked
	// are the callees it calls through their links before they have code,
	// compiled after it too, which native code then calls, the caller's code
	// as it is.
	wanted, linked []*closure
	// keeps is how many keep cells the code uses (ssa.Func.Keeps), and
	// keepLimit how many the build may number (ssa.KeepFeedback), or 0.
	keeps, keepLimit int
	// e is the entry being compiled again, whose failed speculations
	// (jitEntry.failed) are built generic; nil for a first compile.
	e *jitEntry
	// fwd, for a forwarding constructor's inlined frame, is the
	// construction it is inlined at, whose receiver the frame's sites are
	// known from (jitForwardFrame). fwdArgs, for a frame of a function that
	// only forwards its arguments (bytecode's ForwardsArguments), whose
	// code ran and filled its caches, is the construction whose receiver
	// its apply_arguments calls are made with (jitForwardSiteTarget).
	fwd, fwdArgs *jitForwardFrame
	// prog is the function lowered for inlining, its stack depths what
	// Intrinsic reads, made on first use.
	prog *ir.Program
	// targetsBy are the calls e's code learned (jitTargets), by PC, once
	// asked (targetsAt), and targetsOf its lists as they were then.
	targetsBy map[int][]*jitInline
	targetsOf [2][]jitInline
	// selfCalls are the PCs of the function's calls of itself among them,
	// once Resumed asks (selfKnown).
	selfCalls []int32
	selfKnown bool
	// virtualRoots are VirtualRoot's answers, by pool, held until the
	// compile keeps those escape analysis took (keepVirtuals).
	virtualRoots map[uintptr]jitVirtualRoot
	// seeded marks a closure's feedback, inlined where the function makes
	// it (closureInline): its own calls are those the existing tiers met
	// where it has no code (jitCalleeInline).
	seeded bool
}


// mathIntrinsic is the intrinsic a call of o, of n arguments, a plain
// call's where plain says, makes, of either form's code: the realm's Math
// functions of numbers, each made natively, as V8's code makes them:
// Math.pow where exact, and by Go called from there otherwise; Math.max
// and Math.min of any number of them; a rounding, the sign, fround, sqrt,
// abs, imul and clz32 by instructions of their own; every other by Go
// called from there, the VM's very function.
func (fb *jitFeedback) mathIntrinsic(o *Object, n int, plain bool) (ssa.Intrinsic, bool) {
	k := ssa.Intrinsic{Callee: uintptr(unsafe.Pointer(o)), Plain: plain}
	fd := o.fn()
	switch {
	case n == 0:
		// Only Math.random takes none: the next number of the runtime's
		// cache, which the code reads, made now if it is not yet.
		if o != fb.r.randomFn {
			return ssa.Intrinsic{}, false
		}
		k.Op, k.Random = ssa.OpRandomF64, uintptr(unsafe.Pointer(fb.r.mathRandom()))
	case o == fb.r.powFn && n == 2:
		k.Op = ssa.OpPowF64
	case o == fb.r.atan2Fn && n == 2:
		k.Op, k.Go = ssa.OpMathCall, ssa.GoAtan2
	case o == fb.r.hypotFn && n == 2:
		k.Op, k.Go = ssa.OpMathCall, ssa.GoHypot
	case fd == nil || fd.native == nil:
		return ssa.Intrinsic{}, false
	case fd.mathOp == mathMax:
		k.Op, k.Args = ssa.OpMaxF64, n
	case fd.mathOp == mathMin:
		k.Op, k.Args = ssa.OpMinF64, n
	case fd.mathOp == mathImul && n == 2:
		k.Op = ssa.OpMulI32
	case fd.mathOp == mathClz32 && n == 1:
		k.Op = ssa.OpClz32
	case fd.mathOp != 0 && int(fd.mathOp) < len(unaryMath) && n == 1:
		k.Op = jitUnaryMath(unaryMathName(fd.mathOp))
		if k.Op == ssa.OpMathCall {
			k.Go, k.Fn = ssa.GoMath, int(fd.mathOp)
		}
	default:
		return ssa.Intrinsic{}, false
	}
	keep := fb.keep()
	keep.holders = append(keep.holders, o)
	return k, true
}

// jitBigIntIntrinsic makes k the intrinsic of the call at pc of fd, read at
// q, if it is BigInt's that native code makes (bigint.go in ssa): BigInt(x)
// of one argument, plain, BigInt.asUintN(n, x) or asIntN's of a width n
// pushed right after the read, an integer from 0 to 64, as V8 reduces
// those whose width is a constant; and reports whether it is.
func (fb *jitFeedback) jitBigIntIntrinsic(k *ssa.Intrinsic, fd *funcData, plain bool, pc, q int) bool {
	fn := fb.fn
	switch {
	case plain:
		if fd.mathOp != mathBigInt || fn.Code[pc].A != 1 {
			return false
		}
		k.Op, k.Plain = ssa.OpBigOfF64, true
		return true
	case fn.Code[pc].A != 2 || q+1 >= pc:
		return false
	}
	w := fn.Code[q+1]
	if w.Op != bytecode.OpPushInt || int32(w.A) < 0 || int32(w.A) > 64 {
		return false
	}
	// The width stays where it was pushed, the stack above it until the
	// call: no instruction before it takes it.
	p, slot := fb.prog, fb.prog.Maps[q+1].Depth
	for k := q + 2; k <= pc; k++ {
		if p.Maps[k].Depth < slot+1 {
			return false
		}
	}
	k.Op, k.Fn = ssa.OpZextW, int(int32(w.A))
	if fd.mathOp == mathAsIntN {
		k.Op = ssa.OpSextW
	}
	return true
}

// jitMathArgs is the most arguments a call of Math.max or Math.min is made
// natively with, as ssa takes them.
const jitMathArgs = 8

// jitUnaryMath is the op native code makes a Math function of one number
// with, named name: an instruction of its own, or a call of Go
// (ssa.OpMathCall), which roundings are without SSE4.1 (abi.Encoding's
// Round).
func jitUnaryMath(name string) ssa.Op {
	switch name {
	case "sqrt":
		return ssa.OpSqrtF64
	case "abs":
		return ssa.OpAbsF64
	case "sign":
		return ssa.OpSignF64
	case "fround":
		return ssa.OpFroundF64
	}
	if jitEncoding.Round {
		switch name {
		case "floor":
			return ssa.OpFloorF64
		case "ceil":
			return ssa.OpCeilF64
		case "trunc":
			return ssa.OpTruncF64
		case "round":
			return ssa.OpRoundF64
		}
	}
	return ssa.OpMathCall
}

// instanceOfSite is InstanceOf's site of an instanceof of the constructor
// ctor, the value the code reads before it, as either form's code finds it.
func (fb *jitFeedback) instanceOfSite(ctor *Object) (ssa.InstanceOfSite, bool) {
	r := fb.r
	fd := ctor.fn()
	if fd == nil || fd.bound || proxyOf(ctor) != nil {
		// A built-in is one too: sc_Vector is Array.
		return ssa.InstanceOfSite{}, false
	}
	// Its prototype property, made now if it is not yet, is in its table
	// from then on: the site is the constructor's shape and the index there,
	// which no cache keeps for a function (synthesized) -- the operator
	// checks it has this very function.
	r.materializeFunctionProto(ctor)
	var hc propCache
	r.fillPropCache(&hc, ctor, r.hasInstanceAtom, false)
	has, _ := fb.siteOf(r.hasInstanceAtom, false, &hc)
	i := ctor.findOwn(atomPrototype)
	if has.Shape == 0 || has.Holders[0].Object == 0 || i < 0 || ctor.shape == nil || uintptr(unsafe.Pointer(ctor.shape)) != has.Shape ||
		ctor.shape.keys[i].flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 {
		return ssa.InstanceOfSite{}, false
	}
	proto := ssa.PropertySite{Key: uint32(atomPrototype), Shape: has.Shape, Index: int32(i)}
	h := hc.p1
	if hc.p2 != nil {
		h = hc.p2
	}
	if hv := h.vals()[hc.idx]; !hv.IsObject() || hv.Object() != r.hasInstanceFn {
		return ssa.InstanceOfSite{}, false
	}
	k := fb.keep()
	k.holders = append(k.holders, ctor, r.hasInstanceFn)
	return ssa.InstanceOfSite{Ctor: uintptr(unsafe.Pointer(ctor)), HasInstance: has,
		HasInstanceFn: uintptr(unsafe.Pointer(r.hasInstanceFn)), Prototype: proto}, true
}

// jitForwardFrame is the construction a forwarding constructor
// (bytecode.LeafForward) is inlined at: the constructor's function object,
// the pool its receivers come from, and the construction's argument count.
// What its code reads and calls is known from the receiver, as V8 knows a
// property of an object of a known map: this.k, found on the prototype; k's
// apply, the realm's; and the call, of k, inlined with the construction's
// arguments.
type jitForwardFrame struct {
	ctor *Object
	pool *abi.ObjectPool
	argc int
}

// keep is the feedback that keeps what the code holds by address: the
// function's own, for an inlined callee's too.
func (fb *jitFeedback) keep() *jitFeedback {
	if fb.root != nil {
		return fb.root
	}
	return fb
}

// targetsAt is the calls fb.e's code learned at pc (jitTargets), in their
// order there: found from a table by PC made once. Each site's question --
// its native calls, its built-ins' intrinsics, its loop -- had looked
// through every site's targets, a compile's questions as many as its calls
// times its targets.
func (fb *jitFeedback) targetsAt(pc int) []*jitInline {
	if fb.targetsBy == nil {
		fb.targetsBy = map[int][]*jitInline{}
		for _, x := range jitTargets(fb.e) {
			fb.targetsBy[int(x.pc)] = append(fb.targetsBy[int(x.pc)], x)
		}
		fb.targetsOf = [2][]jitInline{fb.e.nativeCalls, fb.e.inlines}
	}
	return fb.targetsBy[pc]
}

// sameList reports whether a and b are one list: the same elements of the
// same array.
func sameList(a, b []jitInline) bool {
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0])
}

// targetsIn is e's targets at pc, in their order: fb.e's from its table
// (targetsAt) while its lists are as they were when the table was made,
// and otherwise -- another entry's, an inlined callee's, or fb.e's since
// learned -- as jitTargets walks them.
func (fb *jitFeedback) targetsIn(e *jitEntry, pc int) []*jitInline {
	if e == fb.e {
		t := fb.targetsAt(pc)
		if sameList(fb.targetsOf[0], e.nativeCalls) && sameList(fb.targetsOf[1], e.inlines) {
			return t
		}
	}
	var t []*jitInline
	for _, x := range jitTargets(e) {
		if int(x.pc) == pc {
			t = append(t, x)
		}
	}
	return t
}


// jitInline is a call jitCallSeen has seen call a function it may inline:
// the call's PC, the function, and how often native code has left the
// inlined callee for Go there.
type jitInline struct {
	pc    int32
	cl    *closure
	obj   *Object
	exits uint32
	// pool, for a construction made natively, is where its objects come
	// from (abi.ObjectPool); via marks a call of Function.prototype.call
	// that calls cl's function (jitCalledVia).
	pool *abi.ObjectPool
	via  bool
	// push marks a call of Array.prototype.push (jitPushes), pop one of
	// Array.prototype.pop (jitPops), charCode one of
	// String.prototype.charCodeAt (jitCharCodes), fromChar one of
	// String.fromCharCode (jitFromCharCodes), substring one of
	// String.prototype's substring, slice, substr or charAt (jitSubstrings).
	push, pop, charCode, fromChar, substring bool
	// forEach marks a call of Array.prototype.forEach, or another of
	// jitLoops, on an array (jitForEaches), cb the closure of the callback
	// it was given and cbObj its function object.
	forEach bool
	cb      *closure
	cbObj   *Object
	// cbLeft marks a loop whose callback, inlined in it, left native code
	// too often (jitInlineLeft): called natively since.
	cbLeft bool
	// proto is a typed array's loop's typed arrays' prototype
	// (jitLoopTyped), nil for an array's.
	proto *Object
	// parseInt marks a plain call of the realm's parseInt with a number
	// (jitParseInts), indexOf a string's or an array's call of one of
	// jitSearches (jitIndexOfs), bigString a BigInt's call of
	// BigInt.prototype.toString (jitBigStrings), leaf a call of a leaf
	// native code made without a frame (jitLeafSeen).
	parseInt, indexOf, bigString, leaf bool
	// byCode marks a function the call met two objects of, closures its
	// maker makes anew at each call: called by its code (jitSameCode).
	byCode bool
	// forward marks a forwarding call's function, f of an
	// apply_arguments, f.apply(x, arguments) (ssa.CallSite's Forward):
	// called natively with the frame's arguments, as many as there are at
	// run time, never inlined there, where they are not known.
	forward bool
	// fwd, for an inlined forwarding constructor's construction
	// (bytecode.LeafForward), is the method it forwards to, as the code was
	// last compiled for (jitForwardMethod).
	fwd *Object
	// parent, for a construction with a class whose constructor is the
	// one a class without one has (jitDerivedBase), is the class it
	// extends, whose constructor cl is: the call checks obj's prototype is
	// still it (ssa.CallSite's Parent).
	parent *Object
	// active, for a super() native code makes, is the class whose
	// constructor's super() it is, whose prototype obj is: the parent,
	// which cl's code -- obj's, or, obj a class with no constructor of its
	// own, parent's -- constructs with (jitSuperTarget).
	active *Object
}

// jitCallsToInline is how often a call leaves native code before it is
// looked at for inlining: a call seldom made is not worth compiling for.
// jitInlineReoptimizations is how many times a function's code is
// compiled again for its calls, apart from jitReoptimizations: a method
// whose calls are on different branches, as Richards' tasks' are, learns
// them at different times.
// jitCallsHot is how often a call leaves native code, its function past
// those compiles, before it is looked at all the same: up to twice as many.
const (
	jitCallsToInline         = 4
	jitInlineReoptimizations = 4
	jitCallsHot              = 128
)

// What jitCallSeen knows of a call (jitEntry.callSites), past its count: that
// it waits for its callee, too long to inline, to be compiled (jitCallSeen),
// that it is made natively, that it is neither that nor inlined, or that it
// is inlined.
const (
	jitCallWaiting = 252
	jitCallNative  = 253
	jitCallDone    = 254
	jitCallInlined = 255
)

// jitCallTargets is how many functions a call may call natively: one it
// calls that is not among them leaves for Go, which makes the call.
const jitCallTargets = 4

// jitInlineExits is how often native code may leave an inlined callee for
// Go before the call stops being inlined: each such exit has Go make the
// call over again.
const jitInlineExits = 16

// jitTarget is a call's target, at pc: the function object, and, for one
// called by its code (jitInline's byCode), its closure.
type jitTarget struct {
	pc     int32
	byCode bool
	obj    *Object
	cl     *closure
}

// jitTargetsOf is what e's calls are decided to make, natively or inlined,
// as a compile from now would be given them (jitEntry's ssaTargets).
func jitTargetsOf(e *jitEntry) []jitTarget {
	if e == nil || len(e.nativeCalls)+len(e.inlines) == 0 {
		return nil
	}
	t := make([]jitTarget, 0, len(e.nativeCalls)+len(e.inlines))
	for _, xs := range [2][]jitInline{e.nativeCalls, e.inlines} {
		for _, x := range xs {
			t = append(t, jitTarget{pc: x.pc, byCode: x.byCode, obj: x.obj, cl: x.cl})
		}
	}
	return t
}

// jitCallRepeats reports whether the call in at pc of f's function, its
// operands on the VM's stack up to sp, which e's code left to Go, calls a
// target an earlier exit there taught e -- one of its decided native
// calls or inlines, the call still decided so -- that the code was not
// compiled for (ssaTargets), while the code waits to be compiled again for
// what it learned: an exit that teaches nothing, which the code would make
// at every such call until then.
func (r *Runtime) jitCallRepeats(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) bool {
	if e == nil || e.ssa == nil || e.invalid || !e.reoptPending() || e.callSites == nil || pc >= len(e.callSites) {
		return false
	}
	if st := e.callSites[pc]; st != jitCallNative && st != jitCallInlined {
		return false
	}
	o, cl := r.jitCalled(f, sp, in)
	if o == nil || cl == nil {
		return false
	}
	is := func(p int32, byCode bool, obj *Object, tcl *closure) bool {
		return int(p) == pc && (obj == o || byCode && tcl != nil && jitSameCode(tcl, cl))
	}
	learned := slices.ContainsFunc(e.nativeCalls, func(x jitInline) bool { return is(x.pc, x.byCode, x.obj, x.cl) }) ||
		slices.ContainsFunc(e.inlines, func(x jitInline) bool { return is(x.pc, x.byCode, x.obj, x.cl) })
	return learned && !slices.ContainsFunc(e.ssaTargets, func(t jitTarget) bool { return is(t.pc, t.byCode, t.obj, t.cl) })
}


// jitNativeCallers is how many callers' entries an entry tells when it
// inlines more (jitEntry.nativeCallers).
const jitNativeCallers = 8




func init() {
	jit.HostCall = jitHostCall
	jitStackHighLowered = jitForgetHighSeen
}

// jitHostCall is Go's side of a call of Go from native code (jit.HostCall,
// ssa.CallSite's Go) by the runtime the context belongs to: it does what
// the call names, its result to the call's keep cell, and reports whether
// it did. What it refuses, the code leaves for Go to make; it may not run
// the script's code, which is the interpreter's then.
func jitHostCall(ctx *abi.Context) bool {
	r := (*Runtime)(ctx.Host)
	if r == nil || ctx.GoKeep >= abi.MaxKeeps {
		return false
	}
	if ssa.GoOp(ctx.GoOp&0xff) == ssa.GoLeaf && jitGoCallWork == nil {
		// A leaf's call, its function checked by the code (jitLeafCall).
		return r.jitLeafCall(ctx)
	}
	if ssa.GoOp(ctx.GoOp&0xff) == ssa.GoCall && jitGoCallWork == nil {
		// The most frequent, asked first; one that runs nothing made
		// without a frame.
		return r.jitMaybePlain(ctx) && r.jitGoCallPlain(ctx) || r.jitGoCall(ctx)
	}
	if jitGoCallWork != nil {
		jitGoCallWork()
	}
	switch ssa.GoOp(ctx.GoOp & 0xff) {
	case ssa.GoHost, ssa.GoArgument, ssa.GoCells, ssa.GoClosure, ssa.GoClose:
		if r.jitRegIsFrameOp(ctx) {
			// Register code's, whose frame is its window, not the stack
			// form's frame these read (zzjit_zregframe.go).
			return r.jitRegFrameOp(ctx)
		}
	}
	switch ssa.GoOp(ctx.GoOp & 0xff) {
	case ssa.GoLeaf:
		return r.jitLeafCall(ctx)
	case ssa.GoAdd:
		return r.jitGoAdd(ctx)
	case ssa.GoArith:
		// An operator of two BigInts that throws nothing.
		op, ok := jitArithOp(ir.Operator(ctx.GoOp >> abi.GoOpIndexShift))
		return ok && r.jitGoBig(ctx, op, jitGoArg(ctx, 0), jitGoArg(ctx, 1))
	case ssa.GoCompare:
		// A relational comparison of primitives, which converts them
		// without calling anything (jitPlainOperand) -- a BigInt too, which
		// compares with anything that is no symbol: its boolean.
		a, b := jitGoArg(ctx, 0), jitGoArg(ctx, 1)
		if !jitComparable(a) || !jitComparable(b) {
			return false
		}
		c, err := r.compare(a, b)
		if err != nil {
			return false
		}
		op := bytecode.OpGe
		switch ir.Operator(ctx.GoOp >> abi.GoOpIndexShift) {
		case ir.Lt:
			op = bytecode.OpLt
		case ir.Le:
			op = bytecode.OpLe
		case ir.Gt:
			op = bytecode.OpGt
		}
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Bool(relationalResult(op, c))
		return true
	case ssa.GoAppend:
		return r.jitGoAppend(ctx)
	case ssa.GoConcat:
		// As concat joins its parts: primitives, converted without running
		// anything, and nothing that throws -- the same refusals as GoAdd's.
		n := int(ctx.GoOp>>abi.GoOpArgcShift) & 0xff
		if n < 1 || n > abi.MaxGoArgs || r.meter != nil {
			return false
		}
		var parts [abi.MaxGoArgs]Value
		total := 0
		for i := range n {
			p := jitGoArg(ctx, i)
			if p.IsBigInt() {
				// Every part is joined to a string (the lowering's
				// stringConcats), a BigInt too.
				if !jitConcatOperand(p) {
					return false
				}
			} else if !jitPlainOperand(p) {
				return false
			}
			if !p.IsString() && !p.IsNumber() {
				s, err := r.toString(p)
				if err != nil {
					return false
				}
				p = Str(s)
			}
			if p.IsString() {
				if total += p.String().length; total > maxStringLength-64 {
					return false
				}
			}
			parts[i] = p
		}
		if n == 1 {
			// A template's part converted (to_string), as the interpreter
			// converts it: a string is itself, which a join had copied.
			p := parts[0]
			if p.IsNumber() {
				p = Str(r.numberString(p.num))
			}
			*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = p
			return true
		}
		s, err := r.joinParts(parts[:n])
		if err != nil || s.length > maxStringLength-64 {
			return false
		}
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Str(s)
		return true
	case ssa.GoRefill:
		// The pool's address, a constant of the code's, which keeps it.
		return r.jitRefillPool(*(**abi.ObjectPool)(unsafe.Pointer(&ctx.GoArgs[0].Ref)))
	case ssa.GoIterResult:
		// A call of next's result object (mir's genNextCall), of the value
		// and the done; its pool, the code's, filled again if it ran out.
		if pool := *(**abi.ObjectPool)(unsafe.Pointer(&ctx.GoArgs[2].Ref)); pool != nil && pool.Count == 0 {
			r.jitRefillPool(pool)
		}
		// A boolean's word, which a NaN's comparison would not tell.
		done := ctx.GoArgs[1].Num == math.Float64bits(True.num)
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Obj(r.iterResult(jitGoArg(ctx, 0), done))
		return true
	case ssa.GoCall:
		return r.jitMaybePlain(ctx) && r.jitGoCallPlain(ctx) || r.jitGoCall(ctx)
	case ssa.GoCells:
		if ctx.GoOp>>abi.GoOpKindShift&0xff == abi.GoClosurePool {
			// Cells native code takes from their pool where it can: run
			// out, filled again, with more.
			jitRefillCells(*(**abi.ObjectPool)(unsafe.Pointer(&ctx.GoArgs[0].Ref)))
		}
		return r.jitMakeCells(ctx)
	case ssa.GoClosure, ssa.GoClose:
		// On the frame whose locals are where native code assigns
		// captured ones: the one Go entered the code with, or a native
		// callee's, which its context stands for (jitCalleeFrame).
		f := r.jitRunFrame(ctx)
		callee := f == nil
		if callee {
			if f = r.jitCalleeFrame(ctx); f == nil {
				return false
			}
		} else {
			// An arrow shares a derived constructor's binding, which a
			// super() native code made has bound in the context alone.
			jitBindThis(f, ctx)
		}
		i := int(ctx.GoOp >> abi.GoOpIndexShift)
		if ctx.GoOp>>abi.GoOpKindShift&0xff == abi.GoClosurePool {
			// One native code takes from its pool where it can, or the
			// cells a scope's end takes (mir's cellsClose): run out, filled
			// again, with more.
			if pool := *(**abi.ObjectPool)(unsafe.Pointer(&ctx.GoArgs[0].Ref)); pool.Cells != 0 {
				jitRefillCells(pool)
			} else {
				r.jitRefillPool(pool)
			}
		}
		if ctx.Cells != nil {
			// Code whose captured locals are in cells of their own.
			return r.jitCellsClosure(ctx, f, i)
		}
		v := Undefined
		if ssa.GoOp(ctx.GoOp&0xff) == ssa.GoClosure {
			if i >= len(f.cl.consts) || callee && !jitCalleeMakes(f, f.cl.consts[i]) {
				return false
			}
			v = Obj(r.makeClosure(f, f.cl.consts[i]))
		} else if len(f.openUpvalues) != 0 {
			r.closeUpvaluesFrom(f, i)
		}
		if callee {
			r.jit.calleeOpen[ctx.Level] = f.openUpvalues
		}
		ctx.Open = uint64(len(f.openUpvalues))
		if ssa.GoOp(ctx.GoOp&0xff) == ssa.GoClosure {
			// A close's writes none, so that one made with every cell
			// taken (ssa's goFrame) shares one.
			*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
		}
		return true
	case ssa.GoNewArray:
		form := int(ctx.GoOp >> abi.GoOpIndexShift)
		if form&ssa.NewArraySort != 0 {
			v, ok := r.jitSortCopy(form, jitGoArg(ctx, 0), jitGoArg(ctx, 1))
			if ok {
				*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
			}
			return ok
		}
		if form&ssa.NewArrayTyped != 0 {
			v, ok := r.jitNewTypedResult(form, jitGoArg(ctx, 0), jitGoArg(ctx, 1))
			if ok {
				*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
			}
			return ok
		}
		// map's or filter's array, of no length, room for the array's in
		// map's: where the array's species is Array's (plainSpecies), as
		// their ArraySpeciesCreate would make it.
		o := r.plainArray(jitGoArg(ctx, 0))
		n := jitGoArg(ctx, 1)
		if o == nil || !n.IsNumber() || !r.plainSpecies(o) {
			return false
		}
		room := int64(0)
		if form&ssa.NewArrayRoom != 0 {
			room = int64(n.Number())
		}
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Obj(r.newArrayOfLength(room))
		return true
	case ssa.GoElemCell:
		o := *(**Object)(unsafe.Pointer(&ctx.GoArgs[0].Ref))
		return o != nil && jitElemAbsent(o, math.Float64frombits(ctx.GoArgs[1].Num))
	case ssa.GoStringAbsent:
		s := *(**String)(unsafe.Pointer(&ctx.GoArgs[0].Ref))
		return s != nil && r.jitStringAbsent(s, math.Float64frombits(ctx.GoArgs[1].Num))
	case ssa.GoIndexOf:
		// A search of a string or an array (jitSearch), what it searches
		// for, and its position if the call gives one.
		form := int(ctx.GoOp >> abi.GoOpIndexShift)
		argv := &r.jit.searchArgs
		args := argv[:1]
		argv[0] = jitGoArg(ctx, 1)
		if form&ssa.IndexOfFrom != 0 {
			args, argv[1] = argv[:2], jitGoArg(ctx, 2)
		}
		v, ok := r.jitSearch(form>>1, jitGoArg(ctx, 0), args)
		*argv = [2]Value{}
		if ok {
			*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
		}
		return ok
	case ssa.GoKeyedWrite:
		// An element's write, of any receiver, by any key (jitKeyedWrite):
		// undefined, the call's result, which nothing reads, as native
		// code's store writes it (mir's keyedStore).
		if !r.jitKeyedWrite(jitGoArg(ctx, 0), jitGoArg(ctx, 1), jitGoArg(ctx, 2), true) {
			return false
		}
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Undefined
		return true
	case ssa.GoBigString:
		// x.toString() of a BigInt, its radix if the call gives one.
		var argv [1]Value
		args := argv[:0]
		if ctx.GoOp>>abi.GoOpIndexShift != 0 {
			args, argv[0] = argv[:1], jitGoArg(ctx, 1)
		}
		v, ok := jitBigString(jitGoArg(ctx, 0), args)
		if ok {
			*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
		}
		return ok
	case ssa.GoIndex:
		v, ok := r.jitKeyedRead(jitGoArg(ctx, 0), jitGoArg(ctx, 1))
		if ok {
			*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
		}
		return ok
	case ssa.GoHost:
		return false
	case ssa.GoArgument:
		v, ok := r.jitArgument(ctx)
		if ok {
			*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
		}
		return ok
	case ssa.GoHas:
		has, ok := r.jitHas(jitGoArg(ctx, 0), jitGoArg(ctx, 1))
		if ok {
			*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Bool(has)
		}
		return ok
	case ssa.GoDelete, ssa.GoDeleteStrict:
		gone, ok := r.jitDelete(jitGoArg(ctx, 0), jitGoArg(ctx, 1), ssa.GoOp(ctx.GoOp&0xff) == ssa.GoDeleteStrict)
		if ok {
			*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Bool(gone)
		}
		return ok
	case ssa.GoElemWrite:
		o := *(**Object)(unsafe.Pointer(&ctx.GoArgs[0].Ref))
		k, v := math.Float64frombits(ctx.GoArgs[1].Num), Float(math.Float64frombits(ctx.GoArgs[2].Num))
		return o != nil && jitStoreElem(o, k, v)
	case ssa.GoElemStore:
		// Any value, by its words, which its source keeps alive.
		o := *(**Object)(unsafe.Pointer(&ctx.GoArgs[0].Ref))
		if ctx.GoOp>>abi.GoOpIndexShift&ssa.StoreFresh != 0 {
			return o != nil && jitAppendFresh(o, math.Float64frombits(ctx.GoArgs[1].Num), jitGoArg(ctx, 2))
		}
		return o != nil && jitStoreElem(o, math.Float64frombits(ctx.GoArgs[1].Num), jitGoArg(ctx, 2))
	case ssa.GoEqual:
		// == or ===, what native code does not compare (OpEqTagged): no
		// conversion equality makes runs code but an object's to a
		// primitive, compared loosely with one not null or undefined.
		a, b := jitGoArg(ctx, 0), jitGoArg(ctx, 1)
		var eq bool
		if ctx.GoOp>>abi.GoOpIndexShift&1 != 0 {
			eq = a.StrictEquals(b)
		} else {
			if a.IsObject() != b.IsObject() && !a.IsNullish() && !b.IsNullish() {
				return false
			}
			var err error
			if eq, err = r.looseEquals(a, b); err != nil {
				return false
			}
		}
		ctx.GoArgs[0].Num = 0
		if eq {
			ctx.GoArgs[0].Num = 1
		}
		return true
	case ssa.GoNumber:
		// An arithmetic operand's number (OpOddballF64): a primitive's,
		// whose conversion runs nothing and does not throw -- a string's,
		// parsed. An object's, a symbol's or a BigInt's, Go makes the
		// operation of after the code leaves.
		v := jitGoArg(ctx, 0)
		if !jitPlainOperand(v) {
			return false
		}
		n, err := r.toNumber(v)
		if err != nil {
			return false
		}
		ctx.GoArgs[0].Num = math.Float64bits(n)
		return true
	case ssa.GoTypedSeen:
		return r.jitTypedSeen(ctx)
	case ssa.GoTypedView:
		return r.jitTypedView(ctx)
	case ssa.GoTypedRead:
		o := *(**Object)(unsafe.Pointer(&ctx.GoArgs[0].Ref))
		v, ok := jitTypedRead(o, math.Float64frombits(ctx.GoArgs[1].Num))
		if ok {
			ctx.GoArgs[0].Num = math.Float64bits(v)
		}
		return ok
	case ssa.GoPoll:
		// A debugger's interrupt may stop and show the frames, which the
		// exit makes; stress (QJS_JIT_STRESS's budget) wants the exits.
		s := r.jit
		if r.debug != nil || r.jitStress.budget != 0 {
			return false
		}
		if idx := int(ctx.Level); ctx.Live == 0 && idx < len(s.goRuns) && s.goRuns[idx].f != nil && s.ssaCtxs[idx] == ctx {
			if e := s.goRuns[idx].e; e != nil && e.reoptPending() && !e.invalid {
				// The run's code learned what to be compiled again for, at a
				// call of Go, say, which an exit would have counted: its
				// loop leaves at the header (abi.ExitPoll), where the
				// stretch's work is counted toward the code's budget
				// (jitLearnedDue) and the code compiled again once it is
				// spent, as V8's back edges spend its interrupt budget.
				// Never leaving, a loop that ran a whole invocation natively
				// -- each of microbench's tests -- counted none of its work,
				// and its code, its calls of a function made anew by Go, was
				// never compiled again.
				return false
			}
		}
		if idx := int(ctx.Level); ctx.Live == 0 && idx < len(s.regRuns) && s.regRuns[idx].f != nil && s.ssaCtxs[idx] == ctx {
			if e := s.regRuns[idx].e; e != nil && e.life != nil && e.life.reopt && !e.invalid {
				// Register code that learned what to be compiled again for
				// (jitRegReopt), at a call of Go its loop goes on after: the
				// loop leaves at its header (abi.ExitPoll), where its work is
				// counted and the code compiled again once that is due, the
				// loop going on in the new code (jitRegLeft), as the stack
				// form's run leaves above. Never leaving, a loop that runs a
				// whole invocation natively had made each construction it
				// learned by Go to the end.
				return false
			}
		}
		old := r.backEdges
		r.backEdges = s.backEdgeBudget()
		s.pollCredit += r.backEdges - old
		s.budgets++
		return r.checkInterruptPoll() == nil
	case ssa.GoRandom:
		// Math.random's cache, found empty: filled again, its next number
		// the result.
		ctx.GoArgs[0].Num = math.Float64bits(r.random())
		return true
	case ssa.GoPow, ssa.GoMath, ssa.GoAtan2, ssa.GoHypot, ssa.GoMod:
		// Two numbers' bits -- or one, and which function of it -- the
		// result's in the first's place.
		x, y := math.Float64frombits(ctx.GoArgs[0].Num), math.Float64frombits(ctx.GoArgs[1].Num)
		var z float64
		switch ssa.GoOp(ctx.GoOp & 0xff) {
		case ssa.GoPow:
			z = jsPow(x, y)
		case ssa.GoMath:
			fn := ctx.GoArgs[1].Num
			if fn == 0 || fn >= uint64(len(unaryMath)) {
				return false
			}
			z = unaryMath[fn](x)
		case ssa.GoAtan2:
			z = fdlibm.Atan2(x, y)
		case ssa.GoHypot:
			z = jsHypot([]float64{x, y})
		case ssa.GoMod:
			z = jsMod(x, y)
		}
		ctx.GoArgs[0].Num = math.Float64bits(z)
		return true
	}
	return false
}

// jitGoRun is what a run of native code in a context runs: its frame,
// entry and code (jitState's goRuns).
type jitGoRun struct {
	f    *frame
	e    *jitEntry
	code *jit.SSACode
	// inlined are the closures code inlines (jitEntry's ssaInlined), whose
	// frames Go makes at an exit inside one (jitUnwindNative): e's lists
	// are another code's once e is compiled again while the run goes on
	// (jitReoptimize), and code's, closed once its native code has left,
	// keep nothing.
	inlined []*closure
	pinned  bool
	// pinnedAt is where its pins start in goPinned: its own code, then
	// that of the native callees whose calls of Go it is suspended in
	// (jitGoCallNested), all released when its native code leaves. serial
	// names those pins: no other run's, nor this run's after they are
	// released, have it (jitState's pinSerial).
	pinnedAt int
	serial   uint64
	// edges and pc are where its last call of Go returned, or its
	// stretch began: the back-edge counter then, net of the polls'
	// budgets, and the PC -- the start of the work its next call of Go
	// is judged against (jitGoCall), in a run on-stack replacement began
	// (osr).
	edges, pc int
	osr       bool
}

// jitGoUnpin ends what calls of Go the run in context idx made: its code,
// which native code has left, may be closed now, and the code retired
// meanwhile that no other call is suspended in is.
func (r *Runtime) jitGoUnpin(idx int) {
	s := r.jit
	s.goRuns[idx].pinned = false
	clear(s.goPinned[s.goRuns[idx].pinnedAt:])
	s.goPinned = s.goPinned[:s.goRuns[idx].pinnedAt]
	if len(s.retired) != 0 {
		r.jitCloseRetired()
	}
}

// jitGoRunAt notes that context idx runs e's code for f, in a stretch
// from pc, with the back-edge counter, net of the polls' budgets, at
// edges, of a run on-stack replacement began if osr.
func (r *Runtime) jitGoRunAt(idx int, f *frame, e *jitEntry, pc, edges int, osr bool) {
	s := r.jit
	if idx >= len(s.goRuns) {
		s.goRuns = append(s.goRuns, make([]jitGoRun, idx+1-len(s.goRuns))...)
	}
	s.goRuns[idx] = jitGoRun{f: f, e: e, code: e.ssa, inlined: e.ssaInlined, edges: edges, pc: pc, osr: osr}
}

// jitGoPending is a call Go made from native code and left native code
// after (jitGoCall): at its exit, Go puts its result, or throws. ctx is the
// level of the code that made it: a run's own, or a native callee's
// (jitGoCallNested), which Go finishes at its exit (jitFinishExit).
type jitGoPending struct {
	set bool
	pc  int
	ctx int
	v   Value
	err error
	// whole marks an instruction Go ran whole (jitRegWhole), which goes on
	// at register pc next, v written to register reg, if not -1.
	whole     bool
	next, reg int
	// step marks v and err as what a loop's step's iterator's next, which
	// native code called (regStepSites), returned or threw: the step's
	// answer is made of them (jitRegStepCalled) once the frame is written.
	step bool
}

// jitRetired is code compiled again or dropped while a call of Go was
// suspended in it, with what it holds (closeSSA).
type jitRetired struct {
	code *jit.SSACode
	keep any
}

// closeSSA closes code, unless a call of Go is suspended in it, which
// returns there: then code and keep, what it holds -- the shapes it
// compares, the objects and pools it reads -- stay until none is.
func (s *jitState) closeSSA(code *jit.SSACode, keep any) error {
	if code != nil && (len(s.lazy) != 0 || slices.Contains(s.goPinned, code)) {
		if len(s.lazy) != 0 {
			s.lazyHeld++
		}
		s.retired = append(s.retired, jitRetired{code, keep})
		return nil
	}
	return code.Close()
}

// jitGoCall makes, for native code, the call at its exit PC, from the
// operands it passed (ssa.GoCall), as an exit there would have Go make it
// (jitCallAt): the operands past the VM's stack's top, which the call
// leaves above them -- not in the frame's own operand slots, where native
// code may read a value it never loaded -- its result in the call's keep
// cell. Native code goes on, as V8's goes on
// after a call of a function it has no code for, rather than leaving. Only
// at a run's own level: a native call's callee has no frame of the VM's
// for a stack trace to show. A call that threw, or after which the code
// is due to be compiled again, or the runtime stopped, is left for Go to
// finish at its exit (goPending).
func (r *Runtime) jitGoCall(ctx *abi.Context) bool {
	s := r.jit
	idx := int(ctx.Level)
	if jitGoInline(ctx) {
		// Inside an inlined callee, which has no frame for what Go would
		// run, nor its PC for what it would learn: the exit makes its
		// frame, and the interpreter the call.
		return false
	}
	if ctx.Live == abi.LiveCall {
		if len(s.regRuns) != 0 {
			if root := r.jitRegRootOf(idx); root >= 0 {
				// A native callee's of register code (zzjit_zregcall.go):
				// an operation that runs nothing made without the levels'
				// frames (jitRegPlainWhole).
				if cl := (*closure)(ctx.Closure); cl != nil && r.jitRegPlainWhole(ctx, cl, cl.reg) {
					return true
				}
				return r.jitRegGoCallNested(ctx, idx, root)
			}
		}
		return false
	}
	if ctx.Live == 0 && idx < len(s.regRuns) && s.regRuns[idx].f != nil && s.ssaCtxs[idx] == ctx {
		// On a register frame (zzjit_zregnative.go).
		return r.jitRegGoCall(ctx, idx)
	}
	return false
}

// jitRunGoProfit accounts for a call of Go that e's code makes at pc in
// run, a run Go entered at a loop by on-stack replacement, and every
// jitSSAProbe such calls judges whether they pay, as jitCalleeProfit
// judges a native callee's: each costs native code what an exit and an
// entry would (jitSSAMinWork) more than the tree tier's own call of the
// built-in, against the work the code did natively since the call before
// -- the back edges taken, at its loops' mean length, or, with none, the
// distance from that call. Judged at the end of its stretches alone
// (jitSSAProfit), such a run, whose one stretch is the loop, was judged
// when the loop was over: a loop of calls of Date.now, of a generator's
// steps, of Date's methods, ran natively to its end, and so again at each
// invocation, the tree tier making none of its calls. Code whose calls do
// not pay is not entered again, at its start nor by on-stack replacement
// (loopSlow), until it is retried (jitRetrySlow); and it reports so, for
// the code to leave. A run entered at the function's start is judged at
// its end, as before: judged at its calls, AWFY's DeltaBlue's callback of
// Plan.execute, whose call of c.execute() Go made until it was decided,
// was no longer entered by Go, and DeltaBlue took 31% longer.
func (r *Runtime) jitRunGoProfit(e *jitEntry, run *jitGoRun, pc int) bool {
	st := &e.ssaStats
	if back := run.edges - (r.backEdges - r.jit.pollCredit); back > 0 {
		st.runWork += uint64(back) * uint64(e.ssaLoop)
	} else if pc > run.pc {
		st.runWork += uint64(pc - run.pc)
	}
	st.runGoCalls++
	if st.runGoCalls%jitSSAProbe == 0 && st.runWork < st.runGoCalls*jitSSAMinWork && !jitCallsNatively(e) {
		e.entrySlow, e.loopSlow = true, true
	}
	return e.loopSlow
}

// jitCallsNatively reports whether e's code calls functions natively: in
// the existing tiers, its callees would run there too, and their work,
// which the back edges count only in part, is native while it runs. Its
// calls of Go are left unjudged: Octane's RegExp's runBlock0, whose calls
// of string methods Go makes, and whose calls of Exec are native, had run
// in the tree, Exec with it, and RegExp took 11% longer.
func jitCallsNatively(e *jitEntry) bool {
	return len(e.nativeCalls) != 0 || slices.Contains(e.callSites, jitCallNative)
}

// jitGoStacked is how many of a call of Go's n operands, for in, are the
// frame's operands: a super()'s first is its this slot, a local.
func jitGoStacked(in bytecode.Instr, n int) int {
	if in.Op == bytecode.OpSuperCall {
		return n - 1
	}
	return n
}

// jitGoPropertyAt reports whether a call of Go, in, may be made at pc of
// fn: a property's read or write only where fn's instruction there is one
// (jitPropertyMade), which reads it; any other call anywhere.
func jitGoPropertyAt(fn *bytecode.Function, pc int, in bytecode.Instr) bool {
	switch in.Op {
	case bytecode.OpSetProp:
		return pc < len(fn.Code) && fn.Code[pc].Op == bytecode.OpSetProp
	case bytecode.OpGetProp:
		return pc < len(fn.Code) && (fn.Code[pc].Op == bytecode.OpGetProp || fn.Code[pc].Op == bytecode.OpGetPropThis || fn.Code[pc].Op == bytecode.OpGetPropUnder)
	}
	if jitcompile.GoOperation(in.Op) {
		// An operation Go makes (abi.GoCallOp), as the instruction there
		// says.
		return pc < len(fn.Code) && fn.Code[pc].Op == in.Op
	}
	return true
}

// jitPropertySeen learns, at a call of Go that reads the property at pc
// of f's function, which native code did not find, what the read met, as
// at an exit: at a megamorphic site's (jitPolySeen); at an object
// pattern's, which has no cache, the megamorphic cache filled for the
// receiver at sp-1, which the next such read finds natively, the
// property's absence too (jitMegaFillProto).
func (r *Runtime) jitPropertySeen(f *frame, e *jitEntry, pc, sp int) {
	switch in := f.cl.fn.Code[pc]; {
	case in.Op == bytecode.OpGetPropUnder:
		if recv := r.stack[sp-1]; recv.IsObject() && int(in.A) < len(f.cl.names) {
			r.jit.megaFill(recv.Object(), f.cl.names[in.A])
			r.jitMegaFillProto(recv.Object(), f.cl.names[in.A])
		}
	case r.jitSiteMegamorphic(f.cl, e, pc):
		// A megamorphic site's read that native code did not find,
		// learned as at an exit (jitPolySeen): the megamorphic cache
		// filled for the receiver's shape, which the next such read
		// finds natively, as V8's miss handler fills its stub cache.
		r.jitPolySeen(f, e, pc, sp, in)
	case in.Op == bytecode.OpGetProp:
		// Native code did not find it, whatever the interpreter's cache
		// holds: where that holds the shapes the code's cases do not, the
		// read hits it, nothing fills, the site never gives up, and each
		// of those shapes is a call of Go at every read. A loop over
		// objects of a dozen shapes made 20,000 such calls a cycle, for
		// as long as it ran.
		if recv := r.stack[sp-1]; recv.IsObject() && int(in.A) < len(f.cl.names) {
			r.jit.megaFill(recv.Object(), f.cl.names[in.A])
		}
	}
}

// jitRunFrame is the frame Go entered the code running in ctx with, or
// nil for code a native call runs in it, which has none of the VM's.
func (r *Runtime) jitRunFrame(ctx *abi.Context) *frame {
	s := r.jit
	idx := int(ctx.Level)
	if ctx.Live != 0 || idx >= len(s.goRuns) || s.ssaCtxs[idx] != ctx {
		return nil
	}
	return s.goRuns[idx].f
}

// jitCalleeFrame is a frame for the native callee running in ctx, which
// has none of the VM's, to make a closure on or close what closures
// captured (ssa.GoClosure, GoClose): its closure, its locals where its
// context has them, its receiver, and the upvalues its closures have
// captured of those locals, which its context keeps (calleeOpen) -- none
// once its context's Open says so, as a native call clears it. Its code
// closes them as it returns, and Go gives them to the frame it makes of
// the callee if it finishes it (jitUnwindNative). Or nil.
func (r *Runtime) jitCalleeFrame(ctx *abi.Context) *frame {
	s := r.jit
	cl := (*closure)(ctx.Closure)
	if ctx.Live != abi.LiveCall || cl == nil || ctx.Locals == nil {
		return nil
	}
	idx := int(ctx.Level)
	if len(s.calleeOpen) < len(s.ssaCtxs) {
		s.calleeOpen = append(s.calleeOpen, make([][]*upvalue, len(s.ssaCtxs)-len(s.calleeOpen))...)
	}
	if ctx.Open == 0 {
		clear(s.calleeOpen[idx])
		s.calleeOpen[idx] = s.calleeOpen[idx][:0]
	}
	f := &s.calleeFrame
	*f = frame{}
	f.cl = cl
	f.locals = unsafe.Slice((*Value)(ctx.Locals), cl.fn.LocalCount)
	f.openUpvalues = s.calleeOpen[idx]
	f.this, f.newTarget = *(*Value)(unsafe.Pointer(&ctx.This)), Undefined
	return f
}

// jitCaptured counts fn's captured locals: the cells of native code whose
// captured locals are in cells of their own (ssa.GoCells).
func jitCaptured(fn *bytecode.Function) int {
	n := 0
	for _, l := range fn.Locals {
		if l.Captured {
			n++
		}
	}
	return n
}

// jitCellsOf is the context's cells (abi.Context's Cells), fn's code's.
func jitCellsOf(ctx *abi.Context, fn *bytecode.Function) []*upvalue {
	return unsafe.Slice((**upvalue)(ctx.Cells), jitCaptured(fn))
}

// jitMakeCells makes, at an entry of code whose captured locals are in
// cells of their own (ir.Program's Cells, ssa.GoCells), those cells, the
// context's, as V8's code allocates a function's context. On the frame Go
// entered the code with, they are open on its slots -- those closures it
// made before took among them -- which it closes as it is popped. For a
// native callee, whose frame Go does not keep, each is a cell of its own,
// closed, holding its parameter's argument or undefined, as the frame's
// slot is at the call: what a closure captures, and native code reads and
// assigns, from then on, and what Go puts in the frame it makes if the
// callee leaves native code (jitReopenCells).
func (r *Runtime) jitMakeCells(ctx *abi.Context) bool {
	if f := r.jitRunFrame(ctx); f != nil {
		fn := f.cl.fn
		n := jitCaptured(fn)
		if n == 0 {
			return false
		}
		cells, room := jitNewCells(n)
		k := 0
		for slot, l := range fn.Locals {
			if !l.Captured {
				continue
			}
			u := openUpvalue(f, slot)
			if u == nil {
				u = &room[k]
				u.slot = &f.locals[slot]
				f.openUpvalues = append(f.openUpvalues, u)
			}
			cells[k] = u
			k++
		}
		ctx.Cells = unsafe.Pointer(unsafe.SliceData(cells))
		ctx.Open = uint64(len(f.openUpvalues))
		return true
	}
	cl := (*closure)(ctx.Closure)
	if ctx.Live != abi.LiveCall || cl == nil || ctx.Locals == nil {
		return false
	}
	fn := cl.fn
	n := jitCaptured(fn)
	if n == 0 {
		return false
	}
	locals := unsafe.Slice((*Value)(ctx.Locals), fn.LocalCount)
	cells, room := jitNewCells(n)
	k := 0
	for slot, l := range fn.Locals {
		if !l.Captured {
			continue
		}
		u := &room[k]
		u.closed = Undefined
		if slot < fn.ParamCount {
			u.closed = locals[slot]
		}
		u.slot = &u.closed
		cells[k] = u
		k++
	}
	ctx.Cells = unsafe.Pointer(unsafe.SliceData(cells))
	return true
}

// jitNewCells is n cells and the array of pointers to them that code whose
// captured locals are in cells of their own reads them through
// (abi.Context's Cells): for up to four, one allocation, as a closure's
// object holds its captured bindings' (newScriptFuncObject) -- the cells
// had been two in five of AWFY DeltaBlue's allocations with the JIT.
func jitNewCells(n int) ([]*upvalue, []upvalue) {
	switch n {
	case 1:
		x := &struct {
			p [1]*upvalue
			u [1]upvalue
		}{}
		return x.p[:], x.u[:]
	case 2:
		x := &struct {
			p [2]*upvalue
			u [2]upvalue
		}{}
		return x.p[:], x.u[:]
	case 3:
		x := &struct {
			p [3]*upvalue
			u [3]upvalue
		}{}
		return x.p[:], x.u[:]
	case 4:
		x := &struct {
			p [4]*upvalue
			u [4]upvalue
		}{}
		return x.p[:], x.u[:]
	}
	return make([]*upvalue, n), make([]upvalue, n)
}

// jitRefillCells fills pool, a pool of cells (abi.ObjectPool's Cells),
// again, with more, if native code ran it out.
func jitRefillCells(pool *abi.ObjectPool) {
	if pool.Count == 0 {
		pool.Size = min(2*max(pool.Size, abi.PoolSize), abi.PoolCapacity)
		jitFillCells(pool)
	}
}

// jitFillCells fills pool with Size blocks of its Cells cells, as
// jitNewCells makes them, each cell's pointer to the value it holds itself
// (upvalue's closed) and the block's pointers to its cells written, which
// native code takes at its function's entries and writes the values of
// (mir's cellsTake).
func jitFillCells(pool *abi.ObjectPool) {
	n := int(pool.Cells)
	for i := range int(pool.Size) {
		cells, room := jitNewCells(n)
		for k := range room {
			room[k].slot = &room[k].closed
			cells[k] = &room[k]
		}
		pool.Objects[i] = unsafe.Pointer(unsafe.SliceData(cells))
	}
	pool.Count = pool.Size
}

// jitCellsClosure is GoClosure and GoClose for code whose captured locals
// are in cells of their own (abi.Context's Cells), on f, the frame Go
// entered it with or a native callee's stand-in (jitCalleeFrame): a
// closure, its constant i, capturing the cells; or a scope's end, from
// local i on, whose captured locals take fresh cells holding what they
// hold -- the closures that captured the old ones keep them, as closing
// an upvalue leaves them, and a loop's next iteration's capture new ones.
// On the frame Go entered, the old ones are closed and the new ones open
// on its slots, as the interpreter's are.
func (r *Runtime) jitCellsClosure(ctx *abi.Context, f *frame, i int) bool {
	fn := f.cl.fn
	cells := jitCellsOf(ctx, fn)
	run := r.jitRunFrame(ctx) == f
	v := Undefined
	if ssa.GoOp(ctx.GoOp&0xff) == ssa.GoClosure {
		if i >= len(f.cl.consts) || !run && !jitCalleeMakes(f, f.cl.consts[i]) {
			return false
		}
		o := r.newClosureIn(f, f.cl.consts[i], cells)
		if fd := o.fn(); fd != nil && atomic.LoadPointer(&fd.closure.fn.VMCode) != nil {
			planCall(fd)
		}
		v = Obj(o)
	} else {
		if run && len(f.openUpvalues) != 0 {
			r.closeUpvaluesFrom(f, i)
		}
		k := 0
		for slot, l := range fn.Locals {
			if !l.Captured {
				continue
			}
			if slot >= i {
				if run {
					// What the cell holds, which may be one native code
					// gave it (mir's cellsClose), not open on the slot.
					f.locals[slot] = *cells[k].slot
					u := &upvalue{slot: &f.locals[slot]}
					f.openUpvalues = append(f.openUpvalues, u)
					cells[k] = u
				} else {
					u := &upvalue{closed: *cells[k].slot}
					u.slot = &u.closed
					cells[k] = u
				}
			}
			k++
		}
	}
	if run {
		ctx.Open = uint64(len(f.openUpvalues))
	}
	if ssa.GoOp(ctx.GoOp&0xff) == ssa.GoClosure {
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
	}
	return true
}

// jitReopenCells gives f, the frame Go made for native code that left it,
// whose captured locals are in cells of their own (abi.Context's Cells),
// those cells: each local's value in the frame's slot, the cell open on it,
// the frame's to close as it is popped -- the closures that captured it
// see what the interpreter assigns from then on. The context keeps them:
// resumed after its call (jitUnwindNative), the code goes on with them.
func jitReopenCells(f *frame, ctx *abi.Context) {
	cells := jitCellsOf(ctx, f.cl.fn)
	k := 0
	for slot, l := range f.cl.fn.Locals {
		if !l.Captured {
			continue
		}
		if u := cells[k]; u.slot != &f.locals[slot] {
			f.locals[slot] = *u.slot
			u.slot = &f.locals[slot]
			f.openUpvalues = append(f.openUpvalues, u)
		}
		k++
	}
}

// jitReopenRunCells gives f, the frame Go entered code with whose captured
// locals are in cells of their own, the cells native code left in ctx, as
// the interpreter's frame has them: a cell native code gave a local at a
// scope's end (mir's cellsClose), closed, is open on its slot again,
// holding what it held, and those it closed, no longer on the slots, are
// the frame's no more.
func jitReopenRunCells(f *frame, ctx *abi.Context) {
	if len(f.locals) != 0 {
		lo, hi := uintptr(unsafe.Pointer(&f.locals[0])), uintptr(unsafe.Pointer(&f.locals[len(f.locals)-1]))
		kept := f.openUpvalues[:0]
		for _, u := range f.openUpvalues {
			if p := uintptr(unsafe.Pointer(u.slot)); p >= lo && p <= hi {
				kept = append(kept, u)
			}
		}
		clear(f.openUpvalues[len(kept):])
		f.openUpvalues = kept
	}
	jitReopenCells(f, ctx)
	ctx.Open = uint64(len(f.openUpvalues))
}

// jitNeedsMaker reports whether the function template c makes is an arrow
// that reads, or whose arrows read, its maker's new.target or super, or
// has a direct eval, which may read either (bytecode.Function's
// ReadsMaker), which a native callee's context does not have
// (jitCalleeFrame).
func jitNeedsMaker(c Value) bool {
	t := c.template()
	return t != nil && t.fn.ReadsMaker
}

// jitCalleeMakes reports whether Go may make the closure c on f, the
// stand-in for a native callee's frame (jitCalleeFrame), which has no
// function object, new.target or this binding of its own: not an arrow
// that reads its maker's new.target or super (jitNeedsMaker), nor one a
// derived constructor makes before its super() binds its this. The arrow
// shares that binding, which super() initializes later; once it has, the
// binding is fixed -- a second super() throws -- and the arrow captures
// its value, as one made after super() in a frame of the VM's reads it
// from the binding it shares. Refused, the operation leaves native code,
// and the frame Go makes of the callee, its this unbound, makes it.
func jitCalleeMakes(f *frame, c Value) bool {
	if jitNeedsMaker(c) {
		return false
	}
	t := c.template()
	return t == nil || t.fn.Kind != bytecode.KindArrow || f.cl.fn.Kind != bytecode.KindDerivedConstructor || !f.this.IsUninitialized()
}

// jitGoPin pins the code the run in context idx runs, until its native code
// leaves (jitGoUnpin), if it has not been yet.
func (r *Runtime) jitGoPin(idx int) {
	s := r.jit
	if run := &s.goRuns[idx]; !run.pinned {
		s.pinSerial++
		run.pinned, run.pinnedAt, run.serial = true, len(s.goPinned), s.pinSerial
		s.goPinned = append(s.goPinned, run.code)
	}
}

// jitEntryOf is the entry of cl's function: the one cl's hint names, or,
// missing that, the cache's, which cl's hint then names, as an entry from
// Go would have it (tryJITAt). A call of Go from a native callee finds
// each level's so, at every call: by the cache's weak key, a tenth of
// RegExp's time went to finding them.
func (r *Runtime) jitEntryOf(cl *closure) *jitEntry {
	s := r.jit
	if e := s.hint(cl); e != nil {
		return e
	}
	e := s.entryFor(cl.fn)
	if e != nil && e.ssa != nil {
		cl.setHint(hintFor(e))
	}
	return e
}

// jitLevels is the frames the last call of Go from a native callee made
// for its levels (jitGoCallNested), which the next call from the same
// levels, or from those of them it shares, takes as they are, but for
// their PCs: in the run whose pins serial names (0 for none), from context
// root on, n of them, above depth. Nothing else makes a frame there while
// native code runs: a call of Go at the run's own level, which makes the
// built-in's there, forgets them (serial 0), as does the run's native code
// leaving; what a call runs makes its frames above them. Any number of
// levels: limited to four, every call of AWFY's Json's parser, its
// readValue native, made a frame for each of its six to twelve levels.
type jitLevels struct {
	serial         uint64
	root, depth, n int
	at             []jitLevelFrame
}

// jitLevelFrame is a level's frame and entry, and the closure, stack base
// and receiver its context had when it was made.
type jitLevelFrame struct {
	f    *frame
	e    *jitEntry
	cl   unsafe.Pointer
	base uint64
	this abi.Slot
	// instance is a derived construction's (abi.Context's Instance),
	// which its frame's this is until its super() binds it: each
	// construction's its own, its This the same unbound value.
	instance abi.Slot
}

// jitGoMark is where the last call of Go a native callee made returned
// (jitGoCallNested): in which run's pins (serial), the back-edge counter
// then, net of the polls' budgets, and its entry's hint -- which names it
// without keeping it -- and PC. hint 0 is none yet.
type jitGoMark struct {
	serial uint64
	edges  int
	hint   uint32
	pc     int
}

// jitCalleeProfit accounts for a call of Go that e's code makes at pc as a
// native callee, in the run whose pins serial names, and every jitSSAProbe
// such calls decides whether its native code pays, as jitSSAProfit decides
// for code Go enters: each call of Go costs what an exit and an entry
// would (jitSSAMinWork), against the work the code did natively since the
// last -- the back edges taken, at its loops' mean length, and the
// distance from the last call it made, at a later PC in the same run. Called natively,
// its calls cost it the same as entered from Go, but its native callers
// pay no entry: called through Go instead (ExitEnter), each call of it
// would cost one. So it pays while its work makes up for its calls of Go
// beyond its calls in (its link's in); if not -- a function of loops that only
// call built-ins, RegExp's run blocks, a call of Go at every few
// instructions -- Go stops entering it and native callers stop calling it
// (notNative), and the existing tiers run it, its calls of built-ins the
// tree tier's own, cheaper than calls of Go from native code.
func (r *Runtime) jitCalleeProfit(e *jitEntry, serial uint64, pc int) {
	s := r.jit
	st := &e.ssaStats
	// The back edges since the last such call, whatever ran them -- the
	// tiers share the counter, so the guess errs toward the native code --
	// and, in the same run, the distance from the call before it.
	m := &s.goMark
	if back := m.edges - (r.backEdges - s.pollCredit); m.hint != 0 && back > 0 {
		st.calleeWork += uint64(back) * uint64(e.ssaLoop)
	}
	if m.serial == serial && m.hint == e.hint && pc > m.pc {
		st.calleeWork += uint64(pc - m.pc)
	}
	st.calleeGoCalls++
	if st.calleeGoCalls%jitSSAProbe != 0 || st.calleeGoCalls <= e.link.in {
		return
	}
	if st.calleeWork < (st.calleeGoCalls-e.link.in)*jitSSAMinWork {
		e.entrySlow = true
		e.notNative, e.nativeRetry = true, 0
		e.link.clearCode()
		e.nativeBackoff = min(e.nativeBackoff+1, jitNativeBackoffs)
	}
}

// jitStoreElem stores v at index n of o as an assignment would, where that
// runs no script and cannot throw (ssa.GoElemWrite): over an element in
// its dense storage, as the interpreter's own fast path does; into a hole
// there, at the array's length or past it, as setElem grows it, when it
// may grow and nothing up its chain could take the store -- a setter, a
// read-only element. Crypto's BigInteger fills its arrays from their ends
// down, and extends them at their lengths: native code had left for Go at
// each such store.
func jitStoreElem(o *Object, n float64, v Value) bool {
	i := uint32(n)
	if float64(i) != n || o.flags&objMappedArguments != 0 {
		return false
	}
	if o.class == ClassTypedArray {
		// What native code does not store: before the view's storage is
		// looked at, which typedWrite does, for native code's next store;
		// a Float16 element. A view out of its buffer, or past its length,
		// leaves for Go. A BigInt typed array's BigInt, its low 64 bits.
		if v.IsBigInt() {
			return jitStoreBigElem(o, i, v)
		}
		return typedWrite(o, i, v)
	}
	if uint(i) < uint(len(o.elemsOf())) && !isHole(o.elemsOf()[i]) {
		o.ownElems()
		o.extOf().elems[i] = v
		return true
	}
	const want = objExtensible | objArrayLengthWritable
	if o.class != ClassArray || o.flags&(want|objHasSparseElements) != want || !o.noIndexKeys() {
		return false
	}
	for p := o.proto(); p != nil; p = p.proto() {
		if p.class != ClassObject && p.class != ClassArray || len(p.elemsOf()) != 0 || !p.noIndexKeys() {
			return false
		}
	}
	// What setValueProp would do: setElem, unless the store is so far past
	// the end that the element is to be a property of its own.
	return o.setElem(i, v)
}

// jitAppendFresh appends v to o, an array a loop of map's or filter's made
// (ssa.StoreFresh), at n, its length, as CreateDataProperty defines it --
// as Go's arrayOut pushes it -- whatever its prototypes hold: the array is
// an Array no code but the loop's has. It refuses any other store.
func jitAppendFresh(o *Object, n float64, v Value) bool {
	if o.class != ClassArray || o.flags&objHasSparseElements != 0 || n != float64(len(o.elemsOf())) {
		return false
	}
	o.extOf().elems = append(o.elemsOf(), v)
	return true
}

// jitTypedRead reads element n of the typed array o where native code did
// not (ssa.GoTypedRead): a Float16 element, or one past what the view's
// copy of its length has where another agent has grown its shared
// buffer. An element that is not a number -- a BigInt, an index the view
// does not have -- it leaves for an exit.
func jitTypedRead(o *Object, n float64) (float64, bool) {
	t, i, ok := typedElemIndex(o, n)
	if !ok {
		return 0, false
	}
	v := t.getElem(i)
	if !v.IsNumber() {
		return 0, false
	}
	return v.num, true
}

// jitUndefinedCell is the cell native code reads an absent element from
// (abi.Encoding's UndefinedCell): undefined, and nothing writes it.
var jitUndefinedCell = Undefined

// jitElemAbsent reports whether o[n] is undefined with nothing to ask
// (ssa.GoElemCell): n an index o has no element at -- past its length, a
// hole -- and neither o nor anything up its chain, each an ordinary object
// or array, has an element or an index of its own, which could hold a
// value or a getter. A lookup table read past what it holds,
// BI_RC[s.charCodeAt(i)], had left native code at each such read.
func jitElemAbsent(o *Object, n float64) bool {
	i := uint32(n)
	if float64(i) != n || i == math.MaxUint32 {
		return false
	}
	if uint(i) < uint(len(o.elemsOf())) && !isHole(o.elemsOf()[i]) {
		return false
	}
	for p := o; p != nil; p = p.proto() {
		if p.class != ClassObject && p.class != ClassArray || p.flags&(objHasSparseElements|objMappedArguments) != 0 || !p.noIndexKeys() {
			return false
		}
		if p != o && len(p.elemsOf()) != 0 {
			return false
		}
	}
	return true
}

// jitStringAbsent reports whether s[n] is undefined with nothing to ask
// (ssa.GoStringAbsent): n an index at or past s's end, which the string
// has no character at, and neither String.prototype -- the realm's, a
// wrapper of the empty string -- nor anything up its chain, each an
// ordinary object or array, has an element or an index of its own, as
// jitElemAbsent asks of an array's chain. As V8 loads an index past a
// string's end in its code once its feedback has met one, under its
// NoElementsProtector: a tokenizer's read one past its text's end,
// PDF.js's Type1 header's, had left native code at each.
func (r *Runtime) jitStringAbsent(s *String, n float64) bool {
	i := uint32(n)
	if float64(i) != n || i == math.MaxUint32 || int64(i) < int64(s.length) {
		return false
	}
	p := r.proto.str
	if p == nil || p.flags&(objHasSparseElements|objMappedArguments) != 0 || len(p.elemsOf()) != 0 || !p.noIndexKeys() {
		return false
	}
	return p.proto() == nil || jitElemAbsent(p.proto(), n)
}

// jitCloseRetired closes the retired code no call of Go is suspended in.
func (r *Runtime) jitCloseRetired() {
	s := r.jit
	if len(s.lazy) != 0 {
		// Native levels whose frames a call of Go put off, whose code is
		// not pinned (jitLazyLevels): until they go on, none is closed.
		return
	}
	kept := s.retired[:0]
	for _, x := range s.retired {
		if slices.Contains(s.goPinned, x.code) {
			kept = append(kept, x)
			continue
		}
		x.code.Close()
	}
	clear(s.retired[len(kept):])
	s.retired = kept
}

// jitConcatAdd is native code's + (ssa.GoAdd) of two strings, or of a
// string and a number of an integer's string, as add makes it -- the
// concatenation, without the conversions and checks add makes first for
// anything else -- and false for any other, or a string too long, which
// add's path then has.
func (r *Runtime) jitConcatAdd(a, b Value) (Value, bool) {
	switch {
	case a.IsString() && b.IsString():
		sa, sb := a.String(), b.String()
		if sa.length+sb.length > maxStringLength-64 {
			return Undefined, false
		}
		// What concat does where no meter counts.
		return Str(sa.Concat(sb)), true
	case a.IsString() && b.IsNumber():
		if s, ok := r.concatInt(a.String(), b.num, false); ok {
			return Str(s), true
		}
	case b.IsString() && a.IsNumber():
		if s, ok := r.concatInt(b.String(), a.num, true); ok {
			return Str(s), true
		}
	}
	return Undefined, false
}

// jitPlainOperand reports whether v is a primitive + takes without a
// conversion that may throw or call anything: not an object, a symbol or a
// BigInt. A string and a number's, at most 25 code units, are below
// maxStringLength's margin unless the string is near it.
func jitPlainOperand(v Value) bool {
	return !v.IsObject() && !v.IsSymbol() && !v.IsBigInt() && (!v.IsString() || v.String().length <= maxStringLength-64)
}

// jitGoBigAdd is GoAdd's sum of two BigInts, a and b, to the call's keep
// cell, where it throws nothing (jitBigArithSafe).
func (r *Runtime) jitGoBigAdd(ctx *abi.Context, a, b Value) bool {
	return r.jitGoBig(ctx, bytecode.OpAdd, a, b)
}

// jitGoBig is op of a and b, BigInts, to the call's keep cell, where it
// throws nothing (jitBigArithSafe), and what of its operands the call
// gives back (jitBigRelease): Go's side of a BigInt's operator from
// native code, ssa.GoArith's, and GoAdd's and GoAppend's of two. A sum, a
// difference or a product -- the operators of BigInts on the heap that
// loops make most -- is checked here and made by bigOp: the checks
// jitBigArithSafe makes, the shortcut jitBigOp tries for two held in their
// values, and the release had each been a call of their own.
func (r *Runtime) jitGoBig(ctx *abi.Context, op bytecode.Op, a, b Value) bool {
	const limit = maxBigIntBits / bits.UintSize
	var v Value
	var err error
	switch {
	case !a.IsBigInt() || !b.IsBigInt() || r.meter != nil:
		return false
	case a.isShortBig() && b.isShortBig() || op != bytecode.OpAdd && op != bytecode.OpSub && op != bytecode.OpMul:
		if !r.jitBigArithSafe(op, a, b) {
			return false
		}
		v, err = r.jitBigOp(op, a, b)
	case op == bytecode.OpMul && jitBigWords(a)+jitBigWords(b) <= limit, op != bytecode.OpMul && max(jitBigWords(a), jitBigWords(b)) < limit:
		v, err = r.bigOp(op, a, b)
	default:
		return false
	}
	if err != nil {
		return false
	}
	if rel := uint8(ctx.GoOp >> abi.GoOpKindShift); rel != 0 {
		r.jitBigRelease(ctx, v, a, b)
	}
	*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
	return true
}

// jitBigOp is op of a and b, two BigInts, as the interpreter makes it,
// without its conversions, which a BigInt needs none of.
func (r *Runtime) jitBigOp(op bytecode.Op, a, b Value) (Value, error) {
	if a.isShortBig() && b.isShortBig() {
		if v, ok := shortArith(op, a, b); ok {
			return v, nil
		}
	}
	switch op {
	case bytecode.OpBitAnd, bytecode.OpBitOr, bytecode.OpBitXor, bytecode.OpShl, bytecode.OpShr:
		v, _, err := r.bigBitwise(op, a, b)
		return v, err
	}
	return r.bigOp(op, a, b)
}

// jitConcatOperand reports whether v, joined to a string, is converted
// without running anything or throwing: a primitive jitPlainOperand takes,
// or a BigInt whose digits are far fewer than a string may have.
func jitConcatOperand(v Value) bool {
	return jitPlainOperand(v) || v.IsBigInt() && jitBigWords(v) <= jitConcatBigWords
}

// jitConcatBigWords bounds the words of a BigInt joined to a string from
// native code: some 300,000 digits.
const jitConcatBigWords = 1 << 14

// jitAddOperand reports whether v, added to other, is converted without
// running anything or throwing (jitPlainOperand): a BigInt too, added to a
// string -- with a number it throws -- its digits and the string's well
// within a string's length.
func jitAddOperand(v, other Value) bool {
	if jitPlainOperand(v) {
		return true
	}
	return other.IsString() && jitConcatOperand(v) && other.String().length < maxStringLength-64-jitConcatBigWords*20
}

// jitComparable reports whether a relational comparison converts v
// without a conversion that may throw or call anything: a primitive, a
// BigInt too, that is no symbol.
func jitComparable(v Value) bool {
	return jitPlainOperand(v) || v.IsBigInt()
}

// jitArithOp is the bytecode operator of ir's arithmetic, bitwise or shift
// operator op.
func jitArithOp(op ir.Operator) (bytecode.Op, bool) {
	switch op {
	case ir.Add:
		return bytecode.OpAdd, true
	case ir.Sub:
		return bytecode.OpSub, true
	case ir.Mul:
		return bytecode.OpMul, true
	case ir.Div:
		return bytecode.OpDiv, true
	case ir.Mod:
		return bytecode.OpMod, true
	case ir.Pow:
		return bytecode.OpPow, true
	case ir.BitAnd:
		return bytecode.OpBitAnd, true
	case ir.BitOr:
		return bytecode.OpBitOr, true
	case ir.BitXor:
		return bytecode.OpBitXor, true
	case ir.Shl:
		return bytecode.OpShl, true
	case ir.Shr:
		return bytecode.OpShr, true
	}
	return 0, false
}

// jitBigArithSafe reports whether op of a and b, both BigInts, throws
// nothing, so that Go makes it from native code (ssa.GoAdd, GoArith):
// making an error may call anything (a getter of Error.stackTraceLimit).
// No result past maxBigIntBits, no division by zero, no memory limit (a
// meter) -- the checks bigArith and bigBitwise make, made before.
func (r *Runtime) jitBigArithSafe(op bytecode.Op, a, b Value) bool {
	if !a.IsBigInt() || !b.IsBigInt() || r.meter != nil {
		return false
	}
	const limit = maxBigIntBits / bits.UintSize
	switch op {
	case bytecode.OpAdd, bytecode.OpSub:
		return max(jitBigWords(a), jitBigWords(b)) < limit
	case bytecode.OpMul:
		return jitBigWords(a)+jitBigWords(b) <= limit
	case bytecode.OpDiv, bytecode.OpMod:
		// A BigInt on the heap is never zero.
		return !b.isInt64Big() || b.int64Big() != 0
	case bytecode.OpPow:
		// Of an exponent that is no negative, and a power as many bits as
		// its base's times it at most, within the limit.
		if !b.isInt64Big() {
			return false
		}
		y := b.int64Big()
		return y >= 0 && y < maxBigIntBits && int64(jitBigWords(a)*bits.UintSize)*y <= maxBigIntBits
	case bytecode.OpBitAnd, bytecode.OpBitOr, bytecode.OpBitXor:
		return true
	case bytecode.OpShl, bytecode.OpShr:
		// Either way a shift by at most what keeps a left shift's result
		// within the limit.
		if !b.isInt64Big() {
			return false
		}
		y := b.int64Big()
		return y > -maxBigIntBits && y < maxBigIntBits && jitBigWords(a)*bits.UintSize+int(max(y, -y)) <= maxBigIntBits
	}
	return false
}

// jitBigWords bounds how many words a BigInt's magnitude has: two for one
// held in its value, of at most 64 bits, which are two words on a 32-bit
// machine. The limit is far enough, maxBigIntBits, that bounding it
// refuses little.
func jitBigWords(v Value) int {
	if v.isShortBig() {
		return 2
	}
	return (*BigInt)(v.ref).digits()
}

// jitGoCallWork, which tests set, runs at each call of Go before it is
// answered: what Go may do there, the collector included.
var jitGoCallWork func()

// jitSSAOptimized, which tests set, is given each function's SSA once it
// is optimized, before the backend compiles it.
var jitSSAOptimized func(fn *bytecode.Function, f *ssa.Func)

// ArrayProtos is ssa.AppendFeedback's: the realm's Array.prototype and
// Object.prototype, as a push the code makes natively checks them, where
// neither has an index in its table (jitStoreElem's walk up the chain,
// which native code's checks of their shapes stand for).
func (fb *jitFeedback) ArrayProtos() ([2]ssa.Holder, bool) {
	ap, op := fb.r.proto.array, fb.r.proto.object
	if ap.shape == nil || op.shape == nil || ap.shape.loose || op.shape.loose || ap.proto() != op || op.proto() != nil ||
		!ap.noIndexKeys() || !op.noIndexKeys() {
		return [2]ssa.Holder{}, false
	}
	k := fb.keep()
	k.shapes = append(k.shapes, remember(ap.shape), remember(op.shape))
	k.holders = append(k.holders, ap, op)
	return [2]ssa.Holder{{Object: uintptr(unsafe.Pointer(ap)), Shape: uintptr(unsafe.Pointer(ap.shape))},
		{Object: uintptr(unsafe.Pointer(op)), Shape: uintptr(unsafe.Pointer(op.shape))}}, true
}

// HasCache is ssa.HasCacheFeedback's: the realm's hasCache, made if it has
// none, which it keeps for as long as it lives, and so for as long as any
// code of its does.
func (fb *jitFeedback) HasCache() uintptr {
	if fb.r.has == nil {
		fb.r.has = new(hasCache)
	}
	return uintptr(unsafe.Pointer(fb.r.has))
}

// KeyAtoms is ssa.KeyAtomsFeedback's: the runtime's keyAtoms, and its
// intStrings, made if it has none, both of which it keeps for as long as it
// lives, and so for as long as any code of its does.
func (fb *jitFeedback) KeyAtoms() (uintptr, uintptr) {
	r := fb.r
	if r.intStrings == nil {
		r.intStrings = new([1024]*String)
	}
	return uintptr(unsafe.Pointer(&r.keyAtoms)), uintptr(unsafe.Pointer(r.intStrings))
}

// keyAtoms' and intStrings' layouts are what native code reads and writes
// (abi.KeyAtomSize).
var _ [0]struct{} = [unsafe.Sizeof(keyAtom{}) - abi.KeyAtomSize]struct{}{}
var _ [0]struct{} = [abi.KeyAtomSize - unsafe.Sizeof(keyAtom{})]struct{}{}
var _ [0]struct{} = [unsafe.Offsetof(keyAtom{}.s) - abi.KeyAtomString]struct{}{}
var _ [0]struct{} = [unsafe.Offsetof(keyAtom{}.atom) - abi.KeyAtomAtom]struct{}{}
var _ [0]struct{} = [abi.KeyAtomAtom - unsafe.Offsetof(keyAtom{}.atom)]struct{}{}
var _ [0]struct{} = [unsafe.Offsetof(keyAtom{}.none) - abi.KeyAtomNone]struct{}{}
var _ [0]struct{} = [abi.KeyAtomNone - unsafe.Offsetof(keyAtom{}.none)]struct{}{}
var _ [0]struct{} = [len(Runtime{}.keyAtoms) - 1<<abi.KeyAtomsBits]struct{}{}
var _ [0]struct{} = [1<<abi.KeyAtomsBits - len(Runtime{}.keyAtoms)]struct{}{}
var _ [0]struct{} = [keySlotMul - abi.KeyAtomsMul]struct{}{}
var _ [0]struct{} = [abi.KeyAtomsMul - keySlotMul]struct{}{}
var _ [0]struct{} = [len(Runtime{}.keyAtoms) - 1<<(64-56)]struct{}{}
var _ [0]struct{} = [atomIndexTag - abi.AtomIndexTag]struct{}{}
var _ [0]struct{} = [abi.AtomIndexTag - atomIndexTag]struct{}{}
var _ [0]struct{} = [len(*Runtime{}.intStrings) - abi.IntStrings]struct{}{}
var _ [0]struct{} = [abi.IntStrings - len(*Runtime{}.intStrings)]struct{}{}

// hasCache's layout is what native code reads (abi.HasEntrySize).
var _ [0]struct{} = [unsafe.Sizeof(hasEntry{}) - abi.HasEntrySize]struct{}{}
var _ [0]struct{} = [abi.HasEntrySize - unsafe.Sizeof(hasEntry{})]struct{}{}
var _ [0]struct{} = [unsafe.Offsetof(hasEntry{}.key) - abi.HasEntryKey]struct{}{}
var _ [0]struct{} = [unsafe.Offsetof(hasEntry{}.shape) - abi.HasEntryShape]struct{}{}
var _ [0]struct{} = [unsafe.Offsetof(hasEntry{}.proto) - abi.HasEntryProto]struct{}{}
var _ [0]struct{} = [unsafe.Offsetof(hasEntry{}.cell) - abi.HasEntryCell]struct{}{}
var _ [0]struct{} = [unsafe.Offsetof(hasEntry{}.has) - abi.HasEntryHas]struct{}{}
var _ [0]struct{} = [hasCacheBits - abi.HasCacheBits]struct{}{}
var _ [0]struct{} = [abi.HasCacheBits - hasCacheBits]struct{}{}
var _ [0]struct{} = [hasCacheMul - abi.HasCacheMul]struct{}{}
var _ [0]struct{} = [abi.HasCacheMul - hasCacheMul]struct{}{}

// WalkedCloseCell is where native code finds the validity cell that says
// closing an array walked in place runs nothing (arrayIterNoReturn): the
// runtime's arrayIterCell, whose first byte is its valid flag. Asked now,
// so that the code finds a cell from its first run if there is one.
func (fb *jitFeedback) WalkedCloseCell() uintptr {
	fb.r.arrayIterNoReturn()
	return uintptr(unsafe.Pointer(&fb.r.arrayIterCell))
}

// A protoCell's valid flag is its first byte, which native code reads
// (WalkedCloseCell).
var _ [0]struct{} = [unsafe.Offsetof(protoCell{}.valid)]struct{}{}

// jitGoCallInstr is the call a call of Go makes (abi.Context's GoOp, which
// native code writes): its instruction, as the bytecode has it at the
// call, and how many operands it takes from the stack. Read from the
// bytecode, it was three loads, one after another, at every call.
func jitGoCallInstr(op uint64) (bytecode.Instr, int) {
	argc := uint32(op>>abi.GoOpArgcShift) & 0xff
	in := bytecode.Instr{Op: bytecode.OpCall, A: argc}
	n := int(argc) + 1
	switch op >> abi.GoOpKindShift & 0xff {
	case abi.GoCallMethod:
		in.Op = bytecode.OpCallMethod
		n++
	case abi.GoCallNew:
		in.Op = bytecode.OpNew
	case abi.GoCallStep:
		// A loop's step, its cursor the one operand.
		return bytecode.Instr{Op: bytecode.OpIterNextOrJump}, 1
	case abi.GoCallFill:
		// A destructuring's step, its cursor the one operand.
		return bytecode.Instr{Op: bytecode.OpIterStep}, 1
	case abi.GoCallForOf:
		// A cursor made, of the iterable, the one operand.
		return bytecode.Instr{Op: bytecode.OpForOfStart}, 1
	case abi.GoCallForIn:
		// A for...in's cursor made, of the object, the one operand.
		return bytecode.Instr{Op: bytecode.OpForInStart}, 1
	case abi.GoCallRest:
		// A destructuring's rest element, of the cursor, the one operand.
		return bytecode.Instr{Op: bytecode.OpIterRest}, 1
	case abi.GoCallGet:
		// A property's read, of the object, which the instruction at the
		// call's PC says (jitPropertyMade).
		return bytecode.Instr{Op: bytecode.OpGetProp, A: uint32(op >> abi.GoOpIndexShift)}, 1
	case abi.GoCallSet:
		// A property's write, of the object and the value.
		return bytecode.Instr{Op: bytecode.OpSetProp, A: uint32(op >> abi.GoOpIndexShift)}, 2
	case abi.GoCallTypeOf:
		return bytecode.Instr{Op: bytecode.OpTypeOf}, 1
	case abi.GoCallInstanceOf:
		return bytecode.Instr{Op: bytecode.OpInstanceOf}, 2
	case abi.GoCallSuper:
		// The frame's this slot, then the arguments.
		return bytecode.Instr{Op: bytecode.OpSuperCall, A: argc}, int(argc) + 1
	case abi.GoCallForward:
		// f.apply(x, arguments), of f, apply and x (ssa.CallSite's
		// Forward): the frame's arguments are f's.
		return bytecode.Instr{Op: bytecode.OpApplyArguments}, 3
	case abi.GoCallOp:
		// An operation's opcode and its A, of its operands: the
		// instruction at the call's PC (jitGoPropertyAt).
		return bytecode.Instr{Op: bytecode.Op(op >> abi.GoOpIndexShift), A: uint32(op >> (abi.GoOpIndexShift + 8))}, int(argc)
	}
	return in, n
}

// jitCallUnsettled reports whether jitCallSeen has anything to note of a
// call of Go at pc: not, once the call is settled as one Go makes
// (jitCallDone) and the code has no inlining pending, unless it constructs
// (a pool to fill). Most calls of Go are such, a built-in's: asked here,
// they cost no call.
func jitCallUnsettled(e *jitEntry, pc int, in bytecode.Instr) bool {
	return in.Op == bytecode.OpNew || e.inlinePending || uint(pc) >= uint(len(e.callSites)) || e.callSites[pc] != jitCallDone
}


// jitClearArgs clears the operands a call of Go put past the VM stack's
// top, from top to sp, so that they keep nothing: a few, each store's
// write barrier tested in line, which clear's call of the runtime costs
// more than.
func jitClearArgs(stack []Value, top, sp int) {
	for i := top; i < sp; i++ {
		stack[i] = Value{}
	}
}

// jitGoArg is a call of Go's operand i (abi.Context's GoArgs).
func jitGoArg(ctx *abi.Context, i int) Value {
	return *(*Value)(unsafe.Pointer(&ctx.GoArgs[i]))
}

// jitInlineLeft counts an exit from the callee inlined at pc, or from its
// call's checks: after jitInlineExits the call is not inlined but made
// natively, if the callee's code allows it, and the code is compiled
// again.
func (r *Runtime) jitInlineLeft(e *jitEntry, pc int) {
	for i := range e.nativeCalls {
		// A loop's callback inlined in it (jitFeedback's callbackInline),
		// called natively once it has left too often.
		if x := &e.nativeCalls[i]; int(x.pc) == pc && x.forEach && !x.cbLeft {
			if x.exits++; x.exits >= jitInlineExits {
				x.cbLeft, e.inlineReopt = true, true
			}
		}
	}
	for i := range e.inlines {
		if x := &e.inlines[i]; int(x.pc) == pc && !slices.Contains(e.notInline, x.pc) {
			if x.exits++; x.exits >= jitInlineExits {
				e.notInline = append(e.notInline, x.pc)
				if e.callSites != nil {
					e.callSites[pc] = jitCallDone
					if x.cl.reg != nil {
						// A construction's with its pool.
						e.nativeCalls = append(e.nativeCalls, jitInline{pc: x.pc, cl: x.cl, obj: x.obj, pool: x.pool})
						e.callSites[pc] = jitCallNative
					}
				}
				e.inlineReopt = true
			}
		}
	}
}

// jitNoInline, where a test sets it, says which functions are not to be
// inlined: those whose native calls, or their code's place in the cache,
// the test is about.
var jitNoInline func(fn *bytecode.Function) bool

// jitInlineDepth is how many levels of calls are inlined: a callee, the
// calls it makes, theirs (ssa's maxInlineDepth).
const jitInlineDepth = 3







// jitViaRead reports whether op, the read of G before G.call(this, ...),
// is one whose value now jitChainValue finds: a global's, a property's of
// what such reads give, or a captured binding's -- the _super a class
// compiled to ES5 calls its parent's constructor through,
// _super.call(this, ...), which had been decided, once, from the call's
// first exit, and left to Go for good where its parent had no code then:
// Typescript's AST constructor called ASTSpan so through Go at each of
// some 40,000 constructions a run, its subclasses theirs.
func jitViaRead(op bytecode.Op) bool {
	switch op {
	case bytecode.OpGetProp, bytecode.OpGetGlobal, bytecode.OpGetUpvalue, bytecode.OpGetUpvalueCheck:
		return true
	}
	return false
}

// jitGoCallFits reports whether native code that calls through a link
// with no code may have Go make the call in, its operands -- a method's
// receiver, the function and the arguments -- no more than a call of Go
// takes (abi.MaxGoArgs), rather than leave native code at it (mir's
// enterExit).
func jitGoCallFits(in bytecode.Instr) bool {
	return int(in.A)+2 <= abi.MaxGoArgs
}

// jitSeedDepth is how many compiles deep seeding a call's target may
// start compiles (jitSeedCallee): a callee's calls' targets, theirs, as
// deep as calls are inlined.
const jitSeedDepth = jitInlineDepth



// jitCallee is a function object a call calls, and its closure.
type jitCallee struct {
	o  *Object
	cl *closure
}


// jitCountAt is how many of calls are at pc.
func jitCountAt(calls []jitInline, pc int) int {
	n := 0
	for _, x := range calls {
		if int(x.pc) == pc {
			n++
		}
	}
	return n
}

// jitSeenCallees are the other functions a method call at pc has called,
// besides o, as its method read's cache saw them -- its entry, those it
// keeps for other shapes (propCache.poly), and those it had before
// (propCache.seen): of each shape it met, the method a prototype held, as
// it holds it now -- the call checks its callee -- each a function
// jitCalled would take. Or none. The poly entries had been left out: a
// polymorphic method of a few classes, Typescript's Symbol's isType and
// isMember, which a subclass overrides, was compiled to call one, and
// learned the other at an exit, one compile later.
func (r *Runtime) jitSeenCallees(cl *closure, p *ir.Program, pc int, o *Object) []jitCallee {
	fn := cl.fn
	if jitCallOp(fn.Code[pc].Op) != bytecode.OpCallMethod {
		return nil
	}
	depth := p.Maps[pc].Depth
	slot := depth - int(fn.Code[pc].A) - 1
	if slot < 1 {
		return nil
	}
	read := -1
	for q := pc - 1; q >= 0; q-- {
		d := p.Maps[q].Depth
		if d == slot && fn.Code[q].Op == bytecode.OpGetPropThis {
			read = q
			break
		}
		if d < slot+1 || d == slot+1 && p.Maps[q+1].Depth < slot+2 {
			return nil
		}
	}
	if read < 0 {
		return nil
	}
	in := fn.Code[read]
	if int(in.B) >= len(cl.ic) || int(in.A) >= len(cl.names) {
		return nil
	}
	var out []jitCallee
	take := func(c *propCache) {
		h := c.p1
		if c.p2 != nil {
			h = c.p2
		}
		if c.shape == nil || c.shape == noShape || c.shape == dictShape || c.getter || h == nil || c.idx < 0 || int(c.idx) >= int(h.valn) {
			return
		}
		prop := h.propAt(c.idx)
		if prop.key != cl.names[in.A] || prop.flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 || !prop.value.IsObject() {
			return
		}
		mo, mc := r.jitCallableClosure(cl, prop.value.Object())
		if mc == nil || mo == o || slices.ContainsFunc(out, func(x jitCallee) bool { return x.o == mo }) {
			return
		}
		out = append(out, jitCallee{mo, mc})
	}
	c := &cl.ic[in.B]
	take(c)
	if p := c.poly; p != nil {
		for i := range p.entries[:p.n] {
			take(&p.entries[i])
		}
	}
	if seen := c.seenOf(); seen != nil {
		for i := range seen.entries[:seen.n] {
			take(&seen.entries[i])
		}
	}
	return out
}

// jitCalleeAt is the function the call or construction at pc calls as the
// existing tiers have met it, and its closure: the read of its callee --
// the nearest before it at the callee's depth, with nothing between but
// the arguments, above it: a method call's get_prop_this, whose cache
// found one plain data property on a prototype, or another's get_global,
// whose cache found the global's, or get_prop of an object so read, a
// namespace's constructor, new Flog.RayTracer.Vector(...), whose value now
// is taken as V8 folds a constant's loads (jitChainValue), or a captured
// binding's read, the function the binding holds now, as V8's code knows
// a closure its context holds -- a function jitCalled would take. Or nil.
// The call checks the function it calls.
func (r *Runtime) jitCalleeAt(cl *closure, p *ir.Program, pc int) (*Object, *closure) {
	o, _ := r.jitCalleeObjectAt(cl, p, pc)
	if o == nil {
		return nil, nil
	}
	return r.jitCallableClosure(cl, o)
}

// jitCalleeObjectAt is the function object jitCalleeAt finds, built-ins'
// too, and the PC of the method read it was found by, or -1.
func (r *Runtime) jitCalleeObjectAt(cl *closure, p *ir.Program, pc int) (*Object, int) {
	fn := cl.fn
	depth := p.Maps[pc].Depth
	slot := depth - int(fn.Code[pc].A) - 1
	op := bytecode.OpGetGlobal
	if jitCallOp(fn.Code[pc].Op) == bytecode.OpCallMethod {
		op = bytecode.OpGetPropThis
	}
	if depth < 0 || slot < 0 || op == bytecode.OpGetPropThis && slot < 1 {
		return nil, -1
	}
	read := -1
	for q := pc - 1; q >= 0; q-- {
		d := p.Maps[q].Depth
		if d == slot && fn.Code[q].Op == op {
			read = q
			break
		}
		if d == slot+1 && op == bytecode.OpGetGlobal && fn.Code[q].Op == bytecode.OpGetProp ||
			d == slot && op == bytecode.OpGetGlobal && (fn.Code[q].Op == bytecode.OpGetUpvalue || fn.Code[q].Op == bytecode.OpGetUpvalueCheck) {
			// A property of an object read so, its value now; or what a
			// captured binding holds now: a module's functions, which call
			// one another through the bindings of the scope they were made
			// in, RegExp's Exec, called by each of runBlock0's two dozen
			// loops, had been decided one call, at its exits, at a time,
			// and the function compiled again for them.
			v, ok := r.jitChainValue(cl, p, q, jitChainReads)
			if !ok || !v.IsObject() {
				return nil, -1
			}
			return v.Object(), -1
		}
		// Only the arguments' instructions, which leave the callee be: the
		// first pushes on it, and the rest work above.
		if d < slot+1 || d == slot+1 && p.Maps[q+1].Depth < slot+2 {
			return nil, -1
		}
	}
	if read < 0 {
		return nil, -1
	}
	in := fn.Code[read]
	if int(in.B) >= len(cl.ic) || int(in.A) >= len(cl.names) {
		return nil, -1
	}
	c := &cl.ic[in.B]
	if op == bytecode.OpGetPropThis && c.p1 == nil {
		// Not found on a prototype: a method read off an object the reads
		// before it name, RayTrace's Flog.RayTracer.Vector.prototype.add(...),
		// its value now (jitChainValue). Calls there had been learned one
		// exit at a time, and its functions ran out of compiles first.
		if v, ok := r.jitChainValue(cl, p, read, jitChainReads); ok && v.IsObject() {
			return v.Object(), read
		}
		return nil, -1
	}
	h, idx := c.p1, c.idx
	if op == bytecode.OpGetGlobal {
		// The global environment, where the read found the name at the
		// cache's index, as pureGlobalSlow looks: not a name a script's
		// lexical binding shadows. Or, where the cache has not found it
		// -- native code reads a global without it, from where the global
		// object had it when compiled (jitFeedback.Global) -- where the
		// environment has it now, as that read finds it: a constructor
		// only native code had read was learned one exit, and one compile,
		// at a time, every construction of DeltaBlue's. A script's
		// lexical binding of the name, a class's say, is where the
		// realm's global scope has it.
		if h = cl.scope(); r.globalLex.valn != 0 && r.lexShadows(cl.names[in.A]) {
			if !r.isGlobalScope(h) {
				return nil, -1
			}
			h, idx = r.globalLex, -1
		}
		if h != nil && (idx < 0 || int(idx) >= int(h.valn) || h.shape.keys[idx].key != cl.names[in.A]) {
			idx = h.findOwn(cl.names[in.A])
		}
	} else {
		if c.shape == nil || c.shape == noShape || c.getter {
			return nil, -1
		}
		if c.p2 != nil {
			h = c.p2
		}
	}
	if h == nil || idx < 0 || int(idx) >= int(h.valn) {
		return nil, -1
	}
	prop := h.propAt(idx)
	if prop.key != cl.names[in.A] || prop.flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 || !prop.value.IsObject() {
		return nil, -1
	}
	return prop.value.Object(), read
}

// jitCallableClosure is o and its closure, if a call in cl's code may make
// it natively or inline it, as jitCalled has a callee; or nil.
func (r *Runtime) jitCallableClosure(cl *closure, o *Object) (*Object, *closure) {
	return r.jitClosureFor(cl, o, false)
}

// jitClosureFor is jitCallableClosure for a call, or for a construction
// (jitNativeTargetOf).
func (r *Runtime) jitClosureFor(cl *closure, o *Object, construct bool) (*Object, *closure) {
	if !r.jitNativeTargetOf(o.fn(), cl, construct) {
		return nil, nil
	}
	return o, o.fn().closure
}

// jitNativeTargetOf reports whether native code of cl's may call fd's
// function, or construct with it: one of the script's, unbound, no
// generator's or async function's template, of cl's realm and scope; and
// a class's constructor only for a construction, as a call of one throws,
// which Go does -- a call, through call or apply too, had run its body.
func (r *Runtime) jitNativeTargetOf(fd *funcData, cl *closure, construct bool) bool {
	return fd != nil && fd.native == nil && !fd.bound && fd.closure != nil && !isGeneratorTemplate(fd.closure.fn) && fd.closure.realm == r.Realm &&
		fd.closure.scope() == cl.scope() && (construct || !isClassConstructorKind(fd.closure.fn.Kind))
}

// jitChainReads is how many reads jitChainValue follows: a global and the
// properties after it, Flog.RayTracer.Vector's three, and the five of
// Flog.RayTracer.Vector.prototype.subtract, a method's, with one to spare.
const jitChainReads = 6

// jitChainValue is the value the read at q of cl's code gives now, if it
// is a global's read, by its cache, or a data property's of an object a
// read just before it gives so, at most n reads in all: a guess, which
// whatever takes it checks.
func (r *Runtime) jitChainValue(cl *closure, p *ir.Program, q, n int) (Value, bool) {
	fn := cl.fn
	if n <= 0 || q < 0 || q >= len(fn.Code) || p.Maps[q].Depth < 0 {
		return Undefined, false
	}
	in := fn.Code[q]
	if in.Op == bytecode.OpGetUpvalue || in.Op == bytecode.OpGetUpvalueCheck {
		// A captured binding's value now, as jitKnownReceiver has it: a
		// constructor a module's closures capture, Box2D's a instanceof S,
		// which had left native code at every test.
		if int(in.A) >= len(cl.upvalues) || cl.upvalues[in.A] == nil {
			return Undefined, false
		}
		if v := cl.upvalues[in.A].get(); !v.IsUninitialized() {
			return v, true
		}
		return Undefined, false
	}
	if int(in.A) >= len(cl.names) {
		return Undefined, false
	}
	switch in.Op {
	case bytecode.OpGetGlobal:
		return r.jitGlobalValue(cl, cl.names[in.A])
	case bytecode.OpGetProp, bytecode.OpGetPropThis:
		recv, ok := r.jitChainReceiver(cl, p, q, n)
		if !ok {
			return Undefined, false
		}
		prop := recv.getOwnVisible(cl.names[in.A])
		if !prop.ok() || prop.flags()&propAccessor != 0 {
			return Undefined, false
		}
		return prop.value(), true
	}
	return Undefined, false
}

// jitGlobalValue is the value of the global name, as a read of it in cl's
// code finds it now: the global object's data property of the name -- or
// the script's lexical binding that shadows it, in the realm's global
// scope -- looked up, not through the read's cache, which native code
// that reads it never fills.
func (r *Runtime) jitGlobalValue(cl *closure, name Atom) (Value, bool) {
	h := cl.scope()
	if h != nil && r.globalLex.valn != 0 && r.lexShadows(name) {
		if !r.isGlobalScope(h) {
			return Undefined, false
		}
		h = r.globalLex
	}
	if h == nil {
		return Undefined, false
	}
	prop := h.getOwnVisible(name)
	if !prop.ok() || prop.flags()&propAccessor != 0 || prop.value().IsUninitialized() {
		return Undefined, false
	}
	return prop.value(), true
}

// jitChainReceiver is the object the property read at q of cl's code reads
// from, if the read just before it gives it so (jitChainValue), at most n
// reads in all: the value that read left on top.
func (r *Runtime) jitChainReceiver(cl *closure, p *ir.Program, q, n int) (*Object, bool) {
	fn := cl.fn
	if n <= 0 || q <= 0 || q >= len(fn.Code) || p.Maps[q].Depth < 0 || p.Maps[q-1].Depth < 0 {
		return nil, false
	}
	switch prev := fn.Code[q-1].Op; {
	case (prev == bytecode.OpGetGlobal || prev == bytecode.OpGetUpvalue || prev == bytecode.OpGetUpvalueCheck) && p.Maps[q-1].Depth == p.Maps[q].Depth-1,
		prev == bytecode.OpGetProp && p.Maps[q-1].Depth == p.Maps[q].Depth:
	default:
		return nil, false
	}
	recv, ok := r.jitChainValue(cl, p, q-1, n-1)
	if !ok || !recv.IsObject() {
		return nil, false
	}
	return recv.Object(), true
}



// jitCalled is the function object a call at the top of the frame's
// operands, up to sp, calls, and its closure, if it is one native code may
// call or inline: one the script made, in the frame's realm and scope.
func (r *Runtime) jitCalled(f *frame, sp int, in bytecode.Instr) (*Object, *closure) {
	at := sp - int(in.A) - 1
	if in.Op == bytecode.OpApplyArguments {
		// f.apply(x, arguments): f, below the realm's apply and x.
		if ap := r.stack[sp-2]; !ap.IsObject() || ap.Object() != r.applyFn {
			return nil, nil
		}
		at = sp - 3
	}
	callee := r.stack[at]
	if !callee.IsObject() {
		return nil, nil
	}
	o := callee.Object()
	if !r.jitNativeTargetOf(o.fn(), f.cl, in.Op == bytecode.OpNew) {
		return nil, nil
	}
	return o, o.fn().closure
}

// jitKnownTargetSeen is jitCallSeen's at a call decided native
// (jitCallNative), with no inlining pending, of a function it calls
// natively already, which has code: jitCallTarget's answers to such a call
// learn nothing -- no new target, no closure of a known function to call by
// its code, no callee to compile -- but count the calls of a callee native
// callers do not call for now (notNative) toward its being called natively
// again (jitRetryNative). It gives that answer, in jitCallTarget's order,
// and reports whether it did; any other call is jitCallSeen's to learn
// from. Such calls are what native code leaves for at each call of a
// callee it calls through Go while the callee is notNative, or whose code
// leaves at its start: asked jitCallSeen's every question, each had cost
// Go a third again of what it costs without them.
func (r *Runtime) jitKnownTargetSeen(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) bool {
	if in.Op == bytecode.OpNew || e.inlinePending || uint(pc) >= uint(len(e.callSites)) || e.callSites[pc] != jitCallNative {
		return false
	}
	o, cl := r.jitCalled(f, sp, in)
	if cl == nil {
		return false
	}
	for i := range e.nativeCalls {
		x := &e.nativeCalls[i]
		if int(x.pc) != pc {
			continue
		}
		if x.cl != nil && x.obj != o && !x.byCode && jitSameCode(x.cl, cl) {
			// Another closure of a function it calls: jitCallTarget's to
			// call by its code.
			return false
		}
		if x.obj == o || x.cl == nil || x.byCode && jitSameCode(x.cl, cl) {
			ce := r.jit.entryFor(cl.fn)
			if (x.obj == o || x.byCode) && ce != nil && ce.notNative {
				jitRetryNative(ce)
				return true
			}
			// One with no code is jitCallTarget's to compile.
			return x.cl == nil || ce != nil && ce.ssa != nil
		}
	}
	return false
}


// jitCalleeAwaited reports whether cl's function has no code for native
// callers only for now: it is being compiled in the background, or would be
// but for the compiles there already (jitNativeCallee), which is no answer
// to whether a call may be native.
func (r *Runtime) jitCalleeAwaited(cl *closure) bool {
	s := r.jit
	if s.jitPending(cl.fn) {
		return true
	}
	if ce := s.entryFor(cl.fn); ce != nil && (ce.ssa != nil || ce.calleeRefused) {
		return false
	}
	return r.jitInBackground(cl.fn) && s.jobsFull()
}



// jitIndexOperands is the receiver and key of the element read in at an
// exit, sp past its operands: a[i]'s, its receiver's kept beneath
// (get_index_this) or not, on the stack; a local's element by a local
// (get_local_index), in the frame. Each is an ir.ArrayRead, which
// jitIndexSeen learns: only a[i]'s had been, so a local's typed array's
// left native code at every invocation -- PDF.js's Huffman tables',
// lengths[val], the rest of each by the interpreter.
func (r *Runtime) jitIndexOperands(f *frame, in bytecode.Instr, sp int) (o, k Value, ok bool) {
	switch in.Op {
	case bytecode.OpGetIndex, bytecode.OpGetIndexThis:
		if sp < 2 || sp > len(r.stack) {
			return Value{}, Value{}, false
		}
		return r.stack[sp-2], r.stack[sp-1], true
	case bytecode.OpGetIndexUnder:
		// An object pattern's computed key's read, A beneath the top.
		if d := int(in.A); sp < d+2 || sp > len(r.stack) {
			return Value{}, Value{}, false
		}
		return r.stack[sp-2-int(in.A)], r.stack[sp-1-int(in.A)], true
	case bytecode.OpGetLocalIndex:
		if int(in.A) >= len(f.locals) || int(in.B) >= len(f.locals) {
			return Value{}, Value{}, false
		}
		return f.locals[in.A], f.locals[in.B], true
	}
	return Value{}, Value{}, false
}

// jitPolyMega reports whether e's code found the site at pc megamorphic
// (jitPolySite's mega).
func jitPolyMega(e *jitEntry, pc int) bool {
	for i := range e.poly {
		if p := &e.poly[i]; int(p.pc) == pc {
			return p.mega
		}
	}
	return false
}

// jitKeyedClass reports whether o's elements by a number Go reads from
// native code where an array's are not (ssa.GoIndex, jitKeyedRead): an
// ordinary object's, and an arguments object's that aliases no parameter
// -- EarleyBoyer's sc_list reads `var a = arguments` so, a[i], which had
// left native code at each read.
func jitKeyedClass(o *Object) bool {
	return o.class == ClassObject || o.class == ClassArguments && o.flags&objMappedArguments == 0
}

// ownEntry is the entry whose code met what fb's function's element reads
// met (jitIndexSeen): the function's own, for an inlined callee too, as an
// inlined callee's property sites are what its own code found. An inlined
// callee's had been the root's alone, nothing: Gameboy's memoryReadNormal,
// parentObj.memory[address] of a typed array, inlined into its callers,
// read it as an array's, which left native code at each read, though its
// own code had learned it.
func (fb *jitFeedback) ownEntry() *jitEntry {
	if fb.root != nil {
		return fb.r.jit.entryFor(fb.fn)
	}
	return fb.e
}

// jitKeyedRead is o[k] where reading it runs nothing (ssa.GoIndex): an
// object's data property by a string key, which the realm's shared
// property cache finds -- no getter, no proxy, no class that answers keys
// itself -- or a dense array's element, or a typed array's, by an
// integer, or an index an ordinary object's table holds or nothing up its
// chain does (jitTableRead); and false for any other, which Go reads
// after an exit.
func (r *Runtime) jitKeyedRead(o, k Value) (Value, bool) {
	if !o.IsObject() {
		return Undefined, false
	}
	obj := o.Object()
	if k.IsNumber() {
		i := uint32(k.Number())
		if float64(i) == k.Number() && uint(i) < uint(len(obj.elemsOf())) && obj.flags&objMappedArguments == 0 {
			if v := obj.elemsOf()[i]; !isHole(v) {
				return v, true
			}
		}
		if t, i, ok := typedElemIndex(obj, k.num); ok {
			return t.getElem(i), true
		}
		if float64(i) != k.Number() || i >= atomIndexTag || obj.class == ClassTypedArray {
			return Undefined, false
		}
		// An index the elements do not hold: a property of an ordinary
		// object's table, as an object used as a map by numbers keeps its
		// keys -- Typescript's caches of type comparisons, by
		// (a.typeID << 16) | b.typeID, dictionaries, which had left native
		// code at every read, the interpreter finishing the invocation --
		// or of none, undefined.
		return r.jitTableRead(obj, Atom(atomIndexTag|i))
	}
	if !k.IsString() {
		return Undefined, false
	}
	// A string's name, found as it is: no conversion runs anything, and no
	// atom is made -- a key no atom has is no property's.
	n, err := r.toPropertyName(k)
	if err != nil {
		return Undefined, false
	}
	a, ok := n.atom, n.hasAtom
	if !ok {
		a, ok = r.atoms.lookup(n.name)
	}
	if !ok {
		// A name no atom has, which no property has: undefined, if
		// nothing up the chain could answer it itself.
		if jitAbsentKey(obj, 0, false) {
			return Undefined, true
		}
		return Undefined, false
	}
	if a.IsIndex() {
		// A canonical index's name, "44": the element its number names,
		// read as by the number -- as V8's keyed loads take a string's
		// array index -- where a table of tokens indexed by the strings a
		// keyword table answers had left native code at every read.
		return r.jitKeyedRead(o, Uint32(a.Index()))
	}
	if v, ok := r.sharedProp(obj, a); ok {
		return v, true
	}
	if jitAbsentKey(obj, a, true) {
		return Undefined, true
	}
	// One the shared cache does not keep: a dictionary's own.
	return r.jitTableRead(obj, a)
}

// jitTableRead is o's property of key a where its chain's tables say what
// it is with nothing run: the plain data property of the nearest object
// whose table, or, for an index, whose elements, have it, or undefined
// where none does -- each object ordinary, its properties its table's and
// none answering the key itself (shapeClass, synthesized), and for an
// index an ordinary object or an array, which have no element they do not
// store (noneIndexed). The global object, whose bindings Go keeps beside
// its table, is not read so. It reports false for any other.
func (r *Runtime) jitTableRead(o *Object, a Atom) (Value, bool) {
	for h, depth := o, 0; h != nil; h, depth = h.proto(), depth+1 {
		if depth > maxChainDepth || !shapeClass(h.class) || h == r.global || r.ensureShape(h) == nil {
			return Undefined, false
		}
		if a.IsIndex() {
			if h.class != ClassObject && h.class != ClassArray {
				return Undefined, false
			}
			if i := a.Index(); uint(i) < uint(len(h.elemsOf())) && !isHole(h.elemsOf()[i]) {
				return h.elemsOf()[i], true
			}
		} else if synthesized(h.class, a) {
			return Undefined, false
		}
		if j := h.findOwn(a); j >= 0 {
			if h.shape.keys[j].flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 {
				return Undefined, false
			}
			return h.vals()[j], true
		}
	}
	return Undefined, true
}

// jitHas is key in obj, as the interpreter answers it, where answering it
// runs nothing and throws nothing (ssa.GoHas): an object none of whose
// chain is a proxy, whose has trap would run, or a module namespace, which
// a key may make evaluate a deferred module, by a key that is no object,
// whose conversion would call its toString or Symbol.toPrimitive. Any
// other -- a right operand that is no object, which throws, included -- it
// refuses (ok false), and Go makes it after an exit (jitHost).
func (r *Runtime) jitHas(key, obj Value) (has, ok bool) {
	if !obj.IsObject() || key.IsObject() {
		return false, false
	}
	o := obj.Object()
	if o.flags&objPolyProto != 0 {
		// An object that keeps its prototype in its table, whose layout
		// says none, matches no entry native code's probe of hasCache
		// reads (mir's keyedHas reads the layout's prototype): asked again
		// by a key hasCache has, it is given a layout of its own holding
		// its prototype, as the tree's in node gives it (hasCached), which
		// the entry hasNote then makes for it matches. Without this a loop
		// asking of such an object called here at every turn.
		if has, ok := hasCached(r, key, o); ok {
			return has, true
		}
	}
	if has, ok := hasQuick(r, key, o); ok {
		// Plain objects and arrays, which hold what they have.
		return has, true
	}
	for p, depth := o, 0; p != nil; p, depth = p.proto(), depth+1 {
		if depth > maxChainDepth || p.class == ClassProxy || p.class == ClassModuleNamespace {
			return false, false
		}
	}
	// A primitive's name, converted with nothing called: a number's, a
	// string's or a symbol's as it is, and a boolean's, null's,
	// undefined's or a BigInt's string.
	n, err := r.toPropertyName(key)
	if err != nil {
		return false, false
	}
	k, known := r.keyFor(o, n)
	if !known {
		// A name no atom has, which no property has.
		return false, true
	}
	has, err = r.hasPropErr(o, k)
	return has, err == nil
}

// jitDelete is delete obj[key], as the interpreter makes it (executeAt's
// delete_prop: toPropertyName, keyFor, deleteProp), where making it runs
// nothing and throws nothing (ssa.GoDelete): an object that is no proxy,
// whose deleteProperty trap would run, no module namespace, which a key
// may make evaluate a deferred module, and none whose own properties a
// class synthesizes -- a typed array's elements, a string wrapper's
// characters -- or ties to a binding -- a mapped arguments object's
// parameters -- by a key that is no object, whose conversion would call
// its toString or Symbol.toPrimitive. An ordinary object's own
// configurable property is removed, its shape changed as the interpreter's
// deletion changes it; an array's element becomes a hole, its length
// unchanged; a property absent is gone already. Any other -- a base that
// is no object, which throws or converts, included -- it refuses (ok
// false), and Go makes it after an exit (jitHost). So does a property
// strict code cannot delete, which throws: deleted as sloppy code deletes
// it, a property that cannot be deleted stays, nothing changed, and the
// interpreter's deletion throws after the exit.
func (r *Runtime) jitDelete(obj, key Value, strict bool) (gone, ok bool) {
	if !obj.IsObject() || key.IsObject() {
		return false, false
	}
	o := obj.Object()
	switch o.class {
	case ClassProxy, ClassModuleNamespace, ClassTypedArray, ClassStringWrapper:
		return false, false
	}
	if o.flags&objMappedArguments != 0 {
		return false, false
	}
	// A primitive's name, converted with nothing called, as jitHas's.
	n, err := r.toPropertyName(key)
	if err != nil {
		return false, false
	}
	k, known := r.keyFor(o, n)
	if !known {
		// A name no atom has, which no property has: deleting it succeeds.
		return true, true
	}
	gone, err = r.deleteProp(o, k, false)
	if err != nil || !gone && strict {
		return false, false
	}
	return gone, true
}

// jitAbsentKey reports whether o's property of a name -- a, or, named
// false, a name no atom has -- is absent from o and from every object up
// its chain, each an ordinary object whose properties are its table's,
// none answering the name itself (synthesized): reading it is undefined,
// with nothing run. A table of strings read by keys it does not have --
// Typescript's StringHashTable, this.table[key] -- had left native code at
// each such read; V8's keyed loads answer a name nothing has through its
// chain's validity cell. Dictionary objects, as such tables become, are
// searched as any are (findOwn).
func jitAbsentKey(o *Object, a Atom, named bool) bool {
	for h, depth := o, 0; h != nil; h, depth = h.proto(), depth+1 {
		if depth > maxChainDepth || !shapeClass(h.class) || named && (synthesized(h.class, a) || h.findOwn(a) >= 0) {
			return false
		}
	}
	return true
}

// jitSameCode reports whether a and b are closures of one function, in one
// environment and realm: a call native code makes of either may check the
// function, and run the closure it finds (ssa.CallSite's ByCode), as each
// call of their maker makes a new one.
func jitSameCode(a, b *closure) bool {
	return a.fn == b.fn && a.env == b.env && a.realm == b.realm
}

// jitReadsArguments reports whether fn uses its arguments object only to
// read it by index, arguments[k] (arguments_index): an object made only on
// demand, never pushed, forwarded to no call (bytecode's
// ForwardsArguments, no apply_arguments), and aliasing no parameter -- a
// sloppy function's with parameters would. Inlined, its reads are the
// call's arguments (ssa.InlineSite's ReadsArguments), as V8 eliminates
// the arguments object; a function that reads arguments had been called
// by Go from native code at each call, its code the tree tier's --
// microbench.js's arguments_test.
func jitReadsArguments(fn *bytecode.Function) bool {
	if !fn.UsesArguments || !fn.ForwardsArguments || fn.MappedArguments && fn.ParamCount > 0 || fn.Leaf == bytecode.LeafForward {
		return false
	}
	return !slices.ContainsFunc(fn.Code, func(in bytecode.Instr) bool { return in.Op == bytecode.OpApplyArguments })
}



// jitForwardMethod is the function object jitForwardTarget's closure is
// of, or nil.
func (r *Runtime) jitForwardMethod(cl *closure, o *Object) *Object {
	fn := cl.fn
	if fn.Leaf != bytecode.LeafForward || len(fn.Code) < 2 || fn.Code[1].Op != bytecode.OpGetProp || int(fn.Code[1].A) >= len(cl.names) {
		return nil
	}
	p := o.getOwnVisible(atomPrototype)
	if !p.ok() || p.flags()&propAccessor != 0 || !p.value().IsObject() {
		return nil
	}
	m := p.value().Object().getOwnVisible(cl.names[fn.Code[1].A])
	if !m.ok() || m.flags()&propAccessor != 0 || !m.value().IsObject() {
		return nil
	}
	t := m.value().Object()
	if !r.jitNativeTargetOf(t.fn(), cl, false) {
		return nil
	}
	return t
}

// forwardProperty is a forwarding constructor's inlined frame's property
// site (jitForwardFrame), known from its receiver: this.k, at its first
// read, as a pool's object finds it; k's apply, at its second, as k does.
func (fb *jitFeedback) forwardProperty(pc int) (ssa.PropertySite, bool) {
	in := fb.fn.Code[pc]
	var recv *Object
	switch {
	case pc == 1 && in.Op == bytecode.OpGetProp:
		pool := fb.fwd.pool
		for i := range pool.Count {
			if o := (*Object)(pool.Objects[i]); o != nil && o.shape != nil {
				recv = o
				break
			}
		}
		if recv == nil {
			// The pool ran out: an object as the next fill makes, whose
			// read the code is compiled for -- else the read would be
			// compiled knowing nothing, and leave every time.
			recv = fb.r.jitPoolObject(fb.fwd.ctor)
		}
	case pc == 2 && in.Op == bytecode.OpGetPropThis:
		recv = fb.r.jitForwardMethod(fb.cl, fb.fwd.ctor)
	}
	site := ssa.PropertySite{Key: uint32(fb.cl.names[in.A])}
	if recv == nil {
		return site, true
	}
	var c propCache
	fb.r.fillPropCache(&c, recv, fb.cl.names[in.A], false)
	return fb.siteFrom(in, &c)
}

// jitUpvalueCells is the address of cl's captured bindings' cells, which
// an inlined frame of its reads them from (ssa.InlineSite's Upvalues), or
// 0 for none.
func jitUpvalueCells(cl *closure) uintptr {
	if len(cl.upvalues) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(unsafe.SliceData(cl.upvalues)))
}

// jitWritesUpvalues reports whether cl's code assigns a binding it
// captured, which an inlined frame of its leaves to Go.
func jitWritesUpvalues(cl *closure) bool {
	return len(cl.upvalues) != 0 && slices.ContainsFunc(cl.fn.Code, func(x bytecode.Instr) bool {
		switch x.Op {
		case bytecode.OpSetUpvalue, bytecode.OpSetUpvalueCheck, bytecode.OpInitUpvalue:
			return true
		}
		return false
	})
}


// jitHasRun reports whether cl's code has run, as a cache of its filled
// says: a function that forwards its arguments is inlined for what its
// caches learned -- its reads of what it forwards to -- which an inlining
// decided before it ran would leave native code at, every construction,
// until its exits gave the inlining up.
func jitHasRun(cl *closure) bool {
	for i := range cl.ic {
		if cl.ic[i].fills != 0 {
			return true
		}
	}
	return false
}


// jitInlineTarget is the call by in at pc of o, cl's function, to inline:
// a construction's with its pool.
func (r *Runtime) jitInlineTarget(pc int32, cl *closure, o *Object, in bytecode.Instr) jitInline {
	x := jitInline{pc: pc, cl: cl, obj: o}
	if in.Op == bytecode.OpNew {
		x.pool = r.jitNewPool(o)
		x.fwd = r.jitForwardMethod(cl, o)
	}
	return x
}

// jitForwardChanged reports whether the forwarding constructor inlined at
// pc in e's code, a construction of o, now forwards to a method other than
// the one the code was compiled for -- its prototype's was replaced -- and
// notes the new one: the code is compiled again for it, as V8 optimizes
// again for a call's new target, rather than the exits counted against
// inlining it (jitInlineLeft).
func (r *Runtime) jitForwardChanged(e *jitEntry, pc int, o *Object) bool {
	for i := range e.inlines {
		x := &e.inlines[i]
		if int(x.pc) != pc || x.obj != o || x.fwd == nil {
			continue
		}
		if t := r.jitForwardMethod(x.cl, o); t != nil && t != x.fwd {
			x.fwd = t
			return true
		}
	}
	return false
}

func (r *Runtime) jitNativeTarget(pc int32, cl *closure, o *Object, in bytecode.Instr) jitInline {
	x := jitInline{pc: pc, cl: cl, obj: o, forward: in.Op == bytecode.OpApplyArguments}
	if in.Op == bytecode.OpNew {
		x.pool = r.jitNewPool(o)
		if fd := o.fn(); fd != nil && fd.closure != cl {
			// A derived class's, constructed with its parent's code
			// (jitDerivedBase).
			x.parent = o.proto()
		}
	}
	return x
}

// jitDerivedBase is the class a class o extends, if o's constructor is the
// one a class without one has -- constructor(...args) { super(...args) },
// which hands its arguments on (bytecode.SuperForward) -- o has no fields
// to give its instances, and native code may construct with the class it
// extends (jitConstructs); or nil. new o(...) then does what a
// construction with that class does, its object made as o makes one --
// o's prototype, o's root shape and room, as constructWithTarget makes it
// for super() to hand on -- and its result o's, as V8 skips a default
// derived constructor (FindNonDefaultConstructorOrConstruct). AWFY's
// Havlak constructs a DictIdEntry, a DictEntry's subclass, for each entry
// of its dictionaries: each had been a call of Go.
func (r *Runtime) jitDerivedBase(o *Object) *Object {
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.native != nil || fd.bound || fd.ctorKind != ctorDerived || fd.fieldInit() != nil ||
		fd.closure.realm != r.Realm || o.class != ClassFunction || o.findOwn(atomPrototype) < 0 || !jitForwardsSuper(fd.closure.fn) {
		return nil
	}
	if p := o.proto(); p != nil && (r.jitConstructs(p) || r.jitConstructsDerived(p)) {
		return p
	}
	return nil
}

// jitConstructsDerived reports whether native code may construct with o, a
// derived class whose constructor is its own -- or a class without one
// whose construction the code of the class it extends cannot make
// (jitDerivedBase), one with fields, or of a parent native code does not
// construct -- called natively as a construction (ssa.CallSite's Derived):
// its instance from a pool made for o, as constructWithTarget makes it
// before the body runs, its super() Go's, which runs the parent and the
// class's field initializers -- a class without a constructor's of the
// call's arguments, which the call writes to its context (jitFrameArgs).
func (r *Runtime) jitConstructsDerived(o *Object) bool {
	fd := o.fn()
	return fd != nil && fd.closure != nil && fd.native == nil && !fd.bound && fd.ctorKind == ctorDerived && fd.superCtor() == o &&
		fd.closure.realm == r.Realm && o.class == ClassFunction && o.findOwn(atomPrototype) >= 0 &&
		(!jitForwardsSuper(fd.closure.fn) || r.jitDerivedBase(o) == nil)
}

// jitForwardsSuper reports whether fn is the constructor a derived class
// without one has: super(...args), its this then the result.
func jitForwardsSuper(fn *bytecode.Function) bool {
	c := fn.Code
	return len(c) == 3 && c[0].Op == bytecode.OpSuperCall && c[0].A == 0 && c[0].B == bytecode.SuperForward &&
		c[1].Op == bytecode.OpDrop && c[2].Op == bytecode.OpReturnUndef
}

// jitCalledVia is the function a method call at the top of the stack
// calls through Function.prototype.call, the realm's own, G.call(this,
// ...): G, its receiver, and its closure, if native code may call it, as
// jitCalled has it; or nil.
func (r *Runtime) jitCalledVia(f *frame, sp int, in bytecode.Instr) (*Object, *closure) {
	if in.Op != bytecode.OpCallMethod || in.A == 0 || r.callFn == nil {
		return nil, nil
	}
	callee, recv := r.stack[sp-int(in.A)-1], r.stack[sp-int(in.A)-2]
	if !callee.IsObject() || callee.Object() != r.callFn || !recv.IsObject() {
		return nil, nil
	}
	o := recv.Object()
	if !r.jitNativeTargetOf(o.fn(), f.cl, false) {
		return nil, nil
	}
	return o, o.fn().closure
}

// jitPops is Array.prototype.pop, the realm's own, if a method call at the
// top of the stack calls it with no argument on an array its fast path
// pops -- dense, of the realm's prototype -- or nil.
func (r *Runtime) jitPops(sp int, in bytecode.Instr) *Object {
	if in.Op != bytecode.OpCallMethod || in.A != 0 {
		return nil
	}
	callee, recv := r.stack[sp-1], r.stack[sp-2]
	if !callee.IsObject() {
		return nil
	}
	fd := callee.Object().fn()
	if fd == nil || fd.elemOp != elemPop || fd.realm != r.Realm || r.plainArray(recv) == nil {
		return nil
	}
	return callee.Object()
}

// jitPushes is Array.prototype.push, the realm's own, if a method call at
// the top of the stack calls it with one argument on an array its fast
// path appends to -- dense, of the realm's prototype, which with its own
// has no element or indexed property (noInheritedIndices) -- or nil.
func (r *Runtime) jitPushes(sp int, in bytecode.Instr) *Object {
	if in.Op != bytecode.OpCallMethod || in.A != 1 {
		return nil
	}
	callee, recv := r.stack[sp-2], r.stack[sp-3]
	if !callee.IsObject() {
		return nil
	}
	fd := callee.Object().fn()
	if fd == nil || fd.elemOp != elemPush || fd.realm != r.Realm {
		return nil
	}
	if o := r.plainArray(recv); o == nil || !r.noInheritedIndices(o) {
		return nil
	}
	return callee.Object()
}

// jitCharCodes is String.prototype.charCodeAt, the realm's own, if a
// method call at the top of the stack calls it with one argument on a
// string, or nil.
func (r *Runtime) jitCharCodes(sp int, in bytecode.Instr) *Object {
	if in.Op != bytecode.OpCallMethod || in.A != 1 || !r.jitCharCodeAt.IsObject() {
		return nil
	}
	callee, recv := r.stack[sp-2], r.stack[sp-3]
	if !callee.IsObject() || callee.Object() != r.jitCharCodeAt.Object() || !recv.IsString() {
		return nil
	}
	return callee.Object()
}

// jitForEaches is Array.prototype.forEach, or another of jitLoops, the
// realm's own, if a method call at the top of the stack calls it on an
// array with one argument, or two, the first a closure of the realm's, and
// that closure and its function object; or nil. One of jitTypedLoops on a
// typed array (jitLoopTyped) is one too, and its prototype the fourth.
func (r *Runtime) jitForEaches(sp int, in bytecode.Instr) (*Object, *closure, *Object, *Object) {
	argc := int(in.A)
	if in.Op != bytecode.OpCallMethod || argc != 1 && argc != 2 {
		return nil, nil, nil, nil
	}
	callee, recv, cb := r.stack[sp-argc-1], r.stack[sp-argc-2], r.stack[sp-argc]
	if !callee.IsObject() || !cb.IsObject() {
		return nil, nil, nil, nil
	}
	var proto *Object
	switch kind := r.jitLoopKind(callee.Object()); {
	case kind == 0:
		return nil, nil, nil, nil
	case kind&abi.LoopTyped != 0:
		t := r.jitLoopTyped(recv)
		if t == nil {
			return nil, nil, nil, nil
		}
		if base := kind & abi.LoopKindMask; t.info().big && base == abi.LoopSort {
			// A BigInt typed array's sort compares BigInts, which its
			// loop does not: Go's.
			return nil, nil, nil, nil
		}
		proto = recv.Object().proto()
	case r.plainArray(recv) == nil:
		return nil, nil, nil, nil
	}
	fd := cb.Object().fn()
	if fd == nil || fd.native != nil || fd.bound || fd.closure == nil || fd.closure.realm != r.Realm {
		return nil, nil, nil, nil
	}
	return callee.Object(), fd.closure, cb.Object(), proto
}



// lexThisSlot is the slot of the frame of fb's function its receiver is
// in, which an arrow it makes captures as its this (closureInline), or -1
// where its code has none (ir.Program's This) or it may not be bound yet:
// a derived constructor's before super().
func (fb *jitFeedback) lexThisSlot() int {
	if fb.prog == nil || !fb.prog.This || fb.fn.Kind == bytecode.KindDerivedConstructor {
		return -1
	}
	return fb.fn.LocalCount + len(fb.fn.Upvalues)
}

// jitCallShape is in's argument count and whether it is a method's call,
// if it is a call, not in tail position.
func jitCallShape(in bytecode.Instr) (int, bool, bool) {
	switch in.Op {
	case bytecode.OpCall:
		return int(in.A), false, true
	case bytecode.OpCallMethod:
		return int(in.A), true, true
	}
	return 0, false, false
}

// madeCallback is the closure the method call at pc of fb's function is
// given, if its code makes it just before the call -- a closure of cl's
// function, made at the instruction before the call, or before the one
// that pushes the loop's this -- which native code inlining the call need
// not make (ssa.MadeCallback): its template, which the code keeps, as the
// inlined frame's closure an exit names, and what it captures. Not an
// arrow that reads its maker's new.target or super (jitNeedsMaker).
func (fb *jitFeedback) madeCallback(pc int, cl *closure) *ssa.MadeCallback {
	for q := pc - 1; q >= 0 && q >= pc-2; q-- {
		in := fb.fn.Code[q]
		if in.Op != bytecode.OpClosure || int(in.A) >= len(fb.cl.consts) {
			continue
		}
		c := jitTemplateCell[struct{}](fb, &fb.cl.consts[in.A])
		t := c.template()
		if t == nil || t.fn != cl.fn || jitNeedsMaker(*c) {
			return nil
		}
		caps := make([]int32, len(t.fn.Upvalues))
		for i, desc := range t.fn.Upvalues {
			caps[i] = int32(desc.Index) << 1
			if desc.FromParent {
				caps[i] |= 1
			}
		}
		k := fb.keep()
		k.inlined = append(k.inlined, t)
		k.made = append(k.made, t)
		return &ssa.MadeCallback{PC: q, Template: uintptr(unsafe.Pointer(c)), Closure: uintptr(unsafe.Pointer(t)), Captures: caps}
	}
	return nil
}

// setLoopSlots notes that e's code for fn holds slots of its loops (ssa's
// forEachCall) n past its frame's operands' first: its native callers make
// its frame that big (jitLink's frame), so that the frames of the callees
// its loops call start past them, as Go's entries do (runSSALoop).
func (e *jitEntry) setLoopSlots(fn *bytecode.Function, n int) {
	e.loopSlots = n
	if l := e.link; l != nil && n > fn.MaxStack {
		l.frame = max(l.frame, uint64(fn.LocalCount+n))
	}
}

// jitGlobalEpochs is the most changes of the global object's bindings
// (shapeTree's globalEpoch) after which a global's read is still compiled
// to check the count alone: a program that keeps deleting globals or
// declaring lexical bindings has each such read check the binding itself,
// rather than leave its code at every change.
const jitGlobalEpochs = 64

// jitLoopDone stands for an exit kind once Go has finished the loop it
// left (jitFinishExit): the frame goes on after the call.
const jitLoopDone = ^uint64(0)

// jitLoopAt is the loop of a built-in's call native code was making in
// ctx, f's context, at its exit at pc, if any (abi.Context's Loop), and
// clears it there: only code that makes such loops (loops, its jitEntry's
// loops) has one, at a method call, which its entries clear.
func jitLoopAt(loops bool, ctx *abi.Context, f *frame, pc int) uint64 {
	loop := jitLoopIn(loops, ctx, f.cl.fn, pc)
	ctx.Loop = 0
	return loop
}

// jitLoopIn is jitLoopAt, the word left as it is: the loop is under way
// still, a level's whose callee is a frame Go makes.
func jitLoopIn(loops bool, ctx *abi.Context, fn *bytecode.Function, pc int) uint64 {
	loop := ctx.Loop
	if loop == 0 || !loops || pc >= len(fn.Code) || jitCallOp(fn.Code[pc].Op) != bytecode.OpCallMethod {
		return 0
	}
	return loop
}

// jitPushLoopFrame pushes, past f's, the frame of the built-in whose call
// f's code makes a loop of (abi.Context's Loop, loop), its callback's
// caller -- as Go's forEach has a frame while its callback runs
// (nativeFrame) -- for the frames Go makes of the loop's callees, which a
// stack trace shows: "at Array.forEach". Its receiver is the loop's array
// where f holds the loop's slots (written, a level Go finishes), or else
// Array.prototype, an array all the same, which the trace names the frame
// by: a level whose call of Go Go makes keeps them in registers, and the
// loop is only ever an array's.
func (r *Runtime) jitPushLoopFrame(f *frame, loop uint64, written bool) {
	kind := int(loop >> 32)
	fe := r.jitLoopFn(kind)
	this := Obj(r.proto.array)
	if t := kind >> abi.LoopTypedShift; kind&abi.LoopTyped != 0 && t < len(r.typedArrayProtos) && r.typedArrayProtos[t] != nil {
		this = Obj(r.typedArrayProtos[t])
	}
	if v := r.stack[f.base+int(loop&0xffffffff)-1]; written && v.IsObject() {
		this = v
	}
	nf := r.pushFrame()
	nf.cl, nf.native, nf.this, nf.newTarget, nf.callee, nf.args = nil, fe.fn().name, this, Undefined, fe, nil
	if len(nf.handlers) != 0 {
		nf.handlers = nf.handlers[:0]
	}
	if len(nf.openUpvalues) != 0 {
		nf.openUpvalues = nf.openUpvalues[:0]
	}
}

// jitMaterialize makes, in f, the frame of e's code that an exit left, the
// closures the code did not make (jitEntry's ssaMade, ssa.MadeCallback):
// a slot that holds the template of one, where the closure would be, gets
// the closure, made as the interpreter's closure instruction makes it on
// the frame -- what it captures the frame's, its cells open on the frame's
// slots once native code has left it -- one for each template, which
// every slot naming it shares, as the code would have made one: as V8's
// deoptimizer materializes the objects escape analysis removed. Go runs
// nothing of the frame before: the exit has written it, its records and
// cells.
func (r *Runtime) jitMaterialize(f *frame, e *jitEntry) {
	r.jitMaterializeBelow(f, e, len(r.stack))
}

// jitMaterializeBelow is jitMaterialize for a frame whose slots end at the
// VM stack's limit: a native caller's level (jitUnwindNative), whose
// callee's frame begins where its operands end. Past them, its homes and
// its loops' slots are its callee's slots -- words naming the callee's
// objects not made, by the callee's homes -- which, read as the caller's,
// had been made of the caller's homes: objects of a shape that was another
// object's field, which the callee then read (RayTrace's Color.addScalar
// under the JIT's stress).
func (r *Runtime) jitMaterializeBelow(f *frame, e *jitEntry, limit int) {
	if e == nil || f.cl == nil {
		return
	}
	if e.ssaHomes != 0 {
		r.jitMaterializeObjects(f, e, limit)
	}
	if len(e.ssaMade) == 0 {
		return
	}
	// The closures this exit makes, from here: those it made at an earlier
	// one are not this one's, though its frame be the same.
	s := r.jit
	clear(s.made)
	s.made = s.made[:0]
	r.jitMaterializeIn(f, f, e, min(max(f.cl.fn.MaxStack, e.loopSlots), limit-f.base))
}

// jitMade is a closure Go made at an exit in place of its template, on the
// frame of the code that did not make it (jitMaterialize).
type jitMade struct {
	f *frame
	t *closure
	o *Object
}

// jitMaterializeIn makes, in f's slots, its locals and the stack slots past
// its base, depth of them, the closures e's code did not make, on the
// frame maker of e's code -- f, or, where f is a callee e's code inlined,
// e's own (jitUnwindNative) -- one for each template, the closures this
// exit made before included (jitState's made): a closure the root gave
// an inlined callee (ssa's calledClosures) is in the callee's slots and
// in the root's.
func (r *Runtime) jitMaterializeIn(f, maker *frame, e *jitEntry, depth int) {
	s := r.jit
	fix := func(v *Value) {
		t := v.template()
		if t == nil || !slices.Contains(e.ssaMade, t) {
			return
		}
		for _, m := range s.made {
			if m.f == maker && m.t == t {
				*v = Obj(m.o)
				return
			}
		}
		o := r.makeClosure(maker, templateValue(t))
		s.made = append(s.made, jitMade{f: maker, t: t, o: o})
		s.materialized++
		*v = Obj(o)
	}
	for i := range f.locals {
		fix(&f.locals[i])
	}
	end := min(len(r.stack), f.base+depth)
	for i := f.base; i < end; i++ {
		fix(&r.stack[i])
	}
}

// jitLoopCallback is the callback of the loop of a built-in's call native
// code was making in f (abi.Context's Loop, loop), as its slots hold it --
// the closure of a callback inlined in it, which Go makes the frame of at
// an exit inside it: the template's, for one the code did not make
// (ssa.MadeCallback), which the frame's slot holds made (jitMaterialize).
func (r *Runtime) jitLoopCallback(f *frame, loop uint64, tmpl *closure) (*closure, *Object) {
	at := f.base + int(loop&0xffffffff) - 1 + 1
	if loop == 0 || at >= len(r.stack) || !r.stack[at].IsObject() {
		panic("jit: an inlined callback's loop has no callback")
	}
	o := r.stack[at].Object()
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.closure.fn != tmpl.fn {
		panic("jit: an inlined callback's loop has another callback")
	}
	return fd.closure, o
}

// jitCalledClosure is the closure f calls at pc, its operands' depth there
// depth: one of tmpl's function the caller's code made and only calls
// (ssa's calledClosures), which it did not make, inlined at the call, whose
// frame Go makes at an exit inside it -- the callee its call's slot holds,
// made by Go in place of the template (jitMaterialize).
func (r *Runtime) jitCalledClosure(f *frame, pc, depth int, tmpl *closure) (*closure, *Object) {
	if pc >= len(f.cl.fn.Code) {
		panic("jit: an inlined closure's call is past its caller's code")
	}
	in := jitCallInstr(f.cl.fn.Code[pc])
	at := f.base + depth - int(in.A) - 1
	if in.Op != bytecode.OpCall || at < f.base || at >= len(r.stack) || !r.stack[at].IsObject() {
		panic("jit: an inlined closure's call has no closure")
	}
	o := r.stack[at].Object()
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.closure.fn != tmpl.fn {
		panic("jit: an inlined closure's call calls another function")
	}
	return fd.closure, o
}

// jitLoopLocals is how many of a loop's slots Go goes on from (ssa's
// loopLocals): the array, the callback, this -- reduce's accumulator --
// the index, the length, the array map or filter makes, the element filter
// or find gave the callback, and its result, uninitialized if none yet.
const jitLoopLocals = 8

// jitSortLocals is how many slots a sort's loop has (ssa's sortLocals):
// the array and the comparator, which Go sorts again from, and the merge
// sort's own, which it clears.
const jitSortLocals = 9

// jitLoopFinish finishes the loop native code was making of a built-in's
// call in f, at pc, at an exit there, as V8's builtin continuation frames
// finish the builtin: loop is the context's Loop word (abi.Context), which says
// which built-in's and where its slots are past f's operands, and after
// whether the loop's call of its callback for the index there was made --
// its callee left native code, and Go finished it, v and err what it
// returned or threw -- or not; or the loop has its result, an exit after
// the call. Go goes on from that index, the callback's result first used
// as the built-in uses it, or from the next one, to the length read as it
// began, as Go's own does (builtins.go): map and filter into the array
// the loop made, an Array whose elements up to the index it appended. f is
// past its call meanwhile, as the interpreter's frame is during one, which
// a stack trace the callback takes shows. The slots keep nothing after.
func (r *Runtime) jitLoopFinish(f *frame, pc int, loop uint64, after bool, v Value, err error) (Value, error) {
	return r.jitLoopFinishAt(f, f.base+int(loop&0xffffffff)-1, uint32(pc+1), loop, after, v, err)
}

// jitLoopFinishAt is jitLoopFinish of the loop whose slots begin at at in
// the VM's stack, f's pc past its call fpc meanwhile: the stack form's
// frame's operands' past its base, a register frame's past its window and
// scratch slots (jitRegLoopFinish).
func (r *Runtime) jitLoopFinishAt(f *frame, at int, fpc uint32, loop uint64, after bool, v Value, err error) (Value, error) {
	slots := r.stack[at : at+jitLoopLocals]
	arr, cb, this, k, n, out, elem, res := slots[0], slots[1], slots[2], slots[3], slots[4], slots[5], slots[6], slots[7]
	clear(slots)
	if err != nil || r.stopped != nil {
		// A throw ends it, as Go's forEach ends at its callback's.
		return v, err
	}
	fe := r.jitLoopFn(int(loop >> 32))
	typed := int(loop>>32)&abi.LoopTyped != 0
	kind := int(loop>>32) & abi.LoopKindMask
	if kind == abi.LoopSort {
		// A sort's exit, anywhere in it: Go sorts the array again from the
		// first, as the built-in sorts it. Nothing the code did shows --
		// its comparator does nothing a call made again could tell apart
		// (jitSortComparator), and the array is as it was until the copy
		// the code sorted is written back, which it is in one call of Go
		// after the last comparison (ssa's sortCall).
		if fe == nil || !arr.IsObject() {
			panic("jit: a sort's slots are not a built-in's")
		}
		clear(r.stack[at+jitLoopLocals : at+jitSortLocals])
		r.jit.loopsFinished++
		r.jit.sortsRestarted++
		f.pc = fpc
		if err := r.nativeFrame(fe, fe.fn(), arr, []Value{cb}, Undefined); err != nil {
			return Undefined, err
		}
		defer func() { r.frameDepth-- }()
		return fe.fn().native(r, arr, []Value{cb})
	}
	makes := kind == abi.LoopMap || kind == abi.LoopFilter
	var t *typedArrayData
	if typed {
		t, _ = arr.Object().dataOf().(*typedArrayData)
	}
	if fe == nil || !arr.IsObject() || !k.IsNumber() || !n.IsNumber() || makes && !out.IsObject() || typed && t == nil {
		panic("jit: a loop's slots are not a built-in's")
	}
	if !after && !res.IsUninitialized() {
		// An exit after the callback's call: its result the loop's.
		after, v = true, res
	}
	from := int64(k.Number())
	r.jit.loopsFinished++
	f.pc = fpc
	// The built-in's frame, as Go's has one (nativeFrame), which a stack
	// trace of what the callback throws shows.
	if err := r.nativeFrame(fe, fe.fn(), arr, []Value{cb, this}, Undefined); err != nil {
		return Undefined, err
	}
	defer func() { r.frameDepth-- }()
	a := &arrayLike{o: arr.Object(), n: int64(n.Number())}
	var o *arrayOut
	var mapped *typedArrayData
	switch {
	case typed && kind == abi.LoopMap:
		// The typed array of the length GoNewArray made, its elements
		// stored at their indices, as %TypedArray%.prototype.map stores
		// them.
		mapped, _ = out.Object().dataOf().(*typedArrayData)
		if mapped == nil {
			panic("jit: a typed array's map has no typed array")
		}
	case makes:
		// The loop's array, of the realm's species, as GoNewArray made it:
		// a typed array's filter's, the elements it kept.
		o = &arrayOut{o: out.Object(), plain: true, n: int64(len(out.Object().elemsOf()))}
	}
	// use is what the built-in does with the callback's result res at index
	// i, of element el: whether it is done, and its result then.
	use := func(i int64, el, res Value) (bool, Value, error) {
		switch kind {
		case abi.LoopReduce:
			this = res
		case abi.LoopMap:
			if mapped != nil {
				// A write past the end of a result that has since shrunk is
				// dropped, as any such write is.
				if err := r.setElem(mapped, int(i), res); err != nil {
					return true, Undefined, err
				}
				break
			}
			if err := o.push(r, res); err != nil {
				return true, Undefined, err
			}
		case abi.LoopFilter:
			if res.Truthy() {
				if err := o.push(r, el); err != nil {
					return true, Undefined, err
				}
			}
		case abi.LoopSome:
			if res.Truthy() {
				return true, True, nil
			}
		case abi.LoopEvery:
			if !res.Truthy() {
				return true, False, nil
			}
		case abi.LoopFind:
			if res.Truthy() {
				return true, el, nil
			}
		case abi.LoopFindIndex:
			if res.Truthy() {
				return true, Float(float64(i)), nil
			}
		}
		return false, Undefined, nil
	}
	if after {
		if done, x, err := use(from, elem, v); done {
			return x, err
		}
		from++
	}
	// As Go's own: the arguments' list made once.
	var argv [4]Value
	for i := from; i < a.n; i++ {
		var el Value
		if typed {
			// As %TypedArray%.prototype's (builtins_typedarray.go): every
			// index, undefined where the view no longer has it.
			el = t.getElem(int(i))
		} else if kind == abi.LoopFind || kind == abi.LoopFindIndex {
			// A hole too, undefined.
			if el, err = a.get(r, i); err != nil {
				return Undefined, err
			}
		} else {
			var present bool
			if el, present, err = a.at(r, i); err != nil {
				return Undefined, err
			}
			if !present {
				if kind == abi.LoopMap {
					// A hole in the array stays a hole in map's.
					if err := o.pushHole(); err != nil {
						return Undefined, err
					}
				}
				continue
			}
		}
		args, recv := argv[:3], this
		argv[0], argv[1], argv[2] = el, Float(float64(i)), arr
		if kind == abi.LoopReduce {
			args, recv = argv[:4], Undefined
			argv[0], argv[1], argv[2], argv[3] = this, el, Float(float64(i)), arr
		}
		res, err := r.callDirect(cb, recv, args)
		if err != nil {
			return Undefined, err
		}
		if done, x, err := use(i, el, res); done {
			return x, err
		}
	}
	switch kind {
	case abi.LoopReduce:
		return this, nil
	case abi.LoopMap, abi.LoopFilter:
		switch {
		case mapped != nil:
			return out, nil
		case typed:
			// The typed array of the elements kept, of the species, as
			// the method makes it after its loop.
			kept := o.o.elemsOf()
			return r.fillTypedArrayLike(arr, t, kept[:min(int(o.n), len(kept))])
		}
		return o.value(), nil
	case abi.LoopSome:
		return False, nil
	case abi.LoopEvery:
		return True, nil
	case abi.LoopFindIndex:
		return Float(-1), nil
	}
	return Undefined, nil
}

// jitNewTypedResult is GoNewArray's for a typed array's map or filter
// (ssa.NewArrayTyped): map's result, a typed array of recv's type and the
// length n, as TypedArraySpeciesCreate makes it; filter's Array of no
// length, which its loop appends the elements it keeps to; and, with
// ssa.NewArrayFill, filter's result made of that Array, kept, as the
// method makes it after its loop. Each only where recv is a typed array a
// loop is made over (jitLoopTyped) whose species is its type's own
// constructor, as it was made (jitPlainTypedSpecies), so that making it
// runs nothing; anything else, and anything that would throw, is refused,
// for Go to make it after an exit.
func (r *Runtime) jitNewTypedResult(form int, recv, n Value) (Value, bool) {
	t := r.jitLoopTyped(recv)
	if t == nil || !r.jitPlainTypedSpecies(recv.Object(), t) {
		return Undefined, false
	}
	// The species its type's own constructor, which makes a view of a
	// buffer of its own, as one of the type is made here: with no frame,
	// which would be written over a native callee's level's, still to be
	// made (jitGoCallNested).
	made := func(length int) (*Object, *typedArrayData, bool) {
		o := newObject(r.typedArrayProtoFor(t.kind), ClassTypedArray)
		nt, err := r.allocTypedArrayChecked(o, t.kind, int64(length))
		return o, nt, err == nil
	}
	switch {
	case form&ssa.NewArrayFill != 0:
		kept := r.plainArray(n)
		if kept == nil {
			return Undefined, false
		}
		els := kept.elemsOf()
		big := t.info().big
		for _, v := range els {
			if big && !v.IsBigInt() || !big && !v.IsNumber() {
				return Undefined, false
			}
		}
		o, nt, ok := made(len(els))
		if !ok {
			return Undefined, false
		}
		for i, v := range els {
			// Numbers, or a BigInt type's BigInts, which convert running
			// nothing.
			if err := r.setElem(nt, i, v); err != nil {
				return Undefined, false
			}
		}
		return Obj(o), true
	case form&ssa.NewArrayRoom != 0:
		if !n.IsNumber() || n.Number() != float64(t.count()) {
			return Undefined, false
		}
		o, _, ok := made(t.count())
		if !ok {
			return Undefined, false
		}
		return Obj(o), true
	}
	return Obj(r.newArrayOfLength(0)), true
}

// jitPlainTypedSpecies reports whether TypedArraySpeciesCreate of o, a
// typed array of t, makes one of its type's own constructor without
// running anything: o has no constructor of its own, its prototype's is
// the type's constructor, a data property, and that constructor's
// Symbol.species is %TypedArray%'s accessor as it was made -- as
// plainSpecies asks of an array.
func (r *Runtime) jitPlainTypedSpecies(o *Object, t *typedArrayData) bool {
	if o.findOwn(atomConstructor) >= 0 || o.proto() == nil {
		return false
	}
	ctor := r.typedArrayCtorFor(t.kind)
	if c := o.proto().getOwn(atomConstructor); !c.ok() || c.flags()&propAccessor != 0 || !c.value().IsObject() || c.value().Object() != ctor {
		return false
	}
	species := r.atoms.internSymbol(r.wellKnown.species)
	if ctor.findOwn(species) >= 0 || ctor.proto() != r.typedArrayCtor || r.jitTypedSpecies == nil {
		return false
	}
	sp := r.typedArrayCtor.getOwn(species)
	if !sp.ok() || sp.flags()&propAccessor == 0 {
		return false
	}
	a := sp.value().accessorPair()
	return a != nil && a.getter == r.jitTypedSpecies
}

// jitSortCopy is GoNewArray's for a sort native code makes (ssa's
// sortCall, ssa.NewArraySort): of recv, a typed array of a number type a
// loop is made over (jitLoopTyped), in bounds, its buffer not immutable --
// one the built-in would throw for is Go's to throw for -- or an Array
// whose every element is a number it has (jitSortArray), the Float64Array
// of twice its length whose first half holds its elements, as numbers --
// and, with ssa.NewArrayFill, x that Float64Array, whose first half is
// written back to recv's elements, in order, the result recv. Anything
// else is refused, before anything has run, for Go to make the call.
func (r *Runtime) jitSortCopy(form int, recv, x Value) (Value, bool) {
	var t *typedArrayData
	// An array's elements are read once, here, not through the array in the
	// loops below: Go 1.27 hoists a.extAt(), an addition to a, out of the
	// loops and their branches, and with a nil -- a typed array's -- that
	// left 0x20 in a pointer's slot, live across the calls, which a
	// collection scanning the stack throws for.
	var elems []Value
	n := 0
	if form&ssa.NewArrayTyped != 0 {
		if t = r.jitLoopTyped(recv); t == nil || t.info().big || t.outOfBounds() || t.storage().immutable {
			return Undefined, false
		}
		n = t.count()
	} else {
		a := r.jitSortArray(recv)
		if a == nil {
			return Undefined, false
		}
		a.ownElems()
		elems = a.elemsOf()
		n = len(elems)
	}
	if form&ssa.NewArrayFill == 0 {
		o := newObject(r.typedArrayProtoFor(elemFloat64), ClassTypedArray)
		s, err := r.allocTypedArrayChecked(o, elemFloat64, 2*int64(n))
		if err != nil {
			return Undefined, false
		}
		b := s.st.bytes[s.byteOffset:s.end]
		for i := range n {
			var v Value
			if t != nil {
				v = t.getElem(i)
			} else {
				v = elems[i]
			}
			binary.LittleEndian.PutUint64(b[i<<3:], math.Float64bits(v.num))
		}
		return Obj(o), true
	}
	if !x.IsObject() {
		return Undefined, false
	}
	s, ok := x.Object().dataOf().(*typedArrayData)
	if !ok || s.kind != elemFloat64 || s.count() != 2*n || s.st == nil {
		// Refused before anything is written: the exit has Go sort the
		// array as it was.
		return Undefined, false
	}
	b := s.st.bytes[s.byteOffset:s.end]
	for i := range n {
		v := Float(math.Float64frombits(binary.LittleEndian.Uint64(b[i<<3:])))
		if t != nil {
			// A number of the array's own, converted back to its type
			// exactly, which no conversion of a number fails.
			if err := r.setElem(t, i, v); err != nil {
				panic("jit: a sorted element not stored back")
			}
		} else {
			elems[i] = v
		}
	}
	return recv, true
}

// jitSortArray is recv if it is an Array a sort native code makes may
// sort (jitSortCopy): of the realm's, extensible, dense (plainArray: its
// length its elements'), whose prototypes have no element
// (noInheritedIndices), its every element a number -- no hole, nothing
// undefined -- so that reading and writing them back is
// what the property protocol would make of them, calling nothing; or nil.
func (r *Runtime) jitSortArray(recv Value) *Object {
	o := r.plainArray(recv)
	if o == nil || o.flags&objExtensible == 0 || !r.noInheritedIndices(o) {
		return nil
	}
	for _, v := range o.elemsOf() {
		if !v.IsNumber() {
			return nil
		}
	}
	return o
}

// jitSortComparator reports whether fn, a sort's comparator, is a body
// native code may call again at will (ssa's sortCall), given numbers: a
// function, an arrow or a method a call runs the body of (DirectCall) --
// no class constructor, which a call refuses -- whose every instruction
// reads a parameter or a constant that is a number, computes with an
// operator, compares, branches forward, or returns: no local of its own,
// no call, no property, no global, no captured binding, no store
// anywhere. Called with numbers, it makes only numbers, booleans,
// undefined and null, runs nothing and changes nothing.
func jitSortComparator(fn *bytecode.Function) bool {
	if fn == nil || !fn.DirectCall || fn.Kind != bytecode.KindNormal && fn.Kind != bytecode.KindArrow && fn.Kind != bytecode.KindMethod ||
		fn.Generator || fn.Async || fn.HasRest || fn.HasDirectEval || !fn.HasSimpleParams || fn.ParamsAreLexical ||
		fn.UsesArguments || fn.UsesThis || fn.LocalCount != fn.ParamCount || len(fn.Code) > 64 {
		return false
	}
	param := func(k uint32) bool { return int(k) < fn.ParamCount }
	operator := func(op bytecode.Op) bool {
		switch op {
		case bytecode.OpAdd, bytecode.OpSub, bytecode.OpMul, bytecode.OpDiv, bytecode.OpMod,
			bytecode.OpBitAnd, bytecode.OpBitOr, bytecode.OpBitXor, bytecode.OpShl, bytecode.OpShr, bytecode.OpUShr,
			bytecode.OpLt, bytecode.OpLe, bytecode.OpGt, bytecode.OpGe, bytecode.OpEq, bytecode.OpNe, bytecode.OpStrictEq, bytecode.OpStrictNe:
			return true
		}
		return false
	}
	for pc, in := range fn.Code {
		switch in.Op {
		case bytecode.OpPushInt, bytecode.OpPushTrue, bytecode.OpPushFalse, bytecode.OpPushUndef, bytecode.OpPushNull,
			bytecode.OpNeg, bytecode.OpNot, bytecode.OpBitNot, bytecode.OpToNumeric, bytecode.OpInc, bytecode.OpDec,
			bytecode.OpDup, bytecode.OpDrop, bytecode.OpSwap, bytecode.OpNop, bytecode.OpReturn, bytecode.OpReturnUndef:
		case bytecode.OpPushConst:
			if int(in.A) >= len(fn.Constants) || fn.Constants[in.A].Kind != bytecode.ConstNumber {
				return false
			}
		case bytecode.OpGetLocal:
			if !param(in.A) {
				return false
			}
		case bytecode.OpGetLocal2:
			if !param(in.A) || !param(in.B) {
				return false
			}
		case bytecode.OpBinLocal:
			if !param(in.A) || !operator(bytecode.Op(in.B)) {
				return false
			}
		case bytecode.OpBinImm:
			if !operator(bytecode.Op(in.B)) {
				return false
			}
		case bytecode.OpLocalBinImm:
			if !param(in.A&(1<<24-1)) || !operator(bytecode.Op(in.A>>24)) {
				return false
			}
		case bytecode.OpJump, bytecode.OpJumpIfFalse, bytecode.OpJumpIfTrue, bytecode.OpJumpIfFalseKeep, bytecode.OpJumpIfTrueKeep:
			if int(in.A) <= pc || int(in.A) > len(fn.Code) {
				return false
			}
		case bytecode.OpJumpIfCmpFalse:
			if int(in.A) <= pc || int(in.A) > len(fn.Code) || !operator(bytecode.Op(in.B)) {
				return false
			}
		default:
			if operator(in.Op) {
				continue
			}
			return false
		}
	}
	return true
}

// jitSearchKind is the kind of o among jitSearches, or -1.
func (r *Runtime) jitSearchKind(o *Object) int {
	if o == nil {
		return -1
	}
	return slices.Index(r.jitSearches[:], o)
}

// jitIndexOfs is one of jitSearches, the realm's own, if a method call at
// the top of the stack calls it with one argument or two on a string --
// String.prototype's -- or an array of the realm's, Array.prototype's; or
// nil.
func (r *Runtime) jitIndexOfs(sp int, in bytecode.Instr) *Object {
	argc := int(in.A)
	if jitCallOp(in.Op) != bytecode.OpCallMethod || argc != 1 && argc != 2 {
		return nil
	}
	callee, recv := r.stack[sp-argc-1], r.stack[sp-argc-2]
	if !callee.IsObject() {
		return nil
	}
	switch k := r.jitSearchKind(callee.Object()); {
	case k < 0:
		return nil
	case k < jitSearchArrays && !recv.IsString(), k >= jitSearchArrays && r.plainArray(recv) == nil:
		return nil
	}
	return callee.Object()
}

// LeafKind is ssa.LeafKindFeedback's: which of the leaves native code
// makes itself where it can the leaf at fn is -- a function Leaf found,
// which the code checks it calls -- by its frameless application (elemOp):
// String.prototype's codePointAt and at, Number's isNaN, isFinite,
// isInteger and isSafeInteger.
func (fb *jitFeedback) LeafKind(fn uintptr) ssa.LeafKind {
	fd := (*Object)(*(*unsafe.Pointer)(unsafe.Pointer(&fn))).fn()
	if fd == nil {
		return ssa.LeafOther
	}
	switch fd.elemOp {
	case elemStrCodePointAt:
		return ssa.LeafCodePointAt
	case elemStrAt:
		return ssa.LeafAt
	case elemNumIsNaN:
		return ssa.LeafNumberIsNaN
	case elemNumIsFinite:
		return ssa.LeafNumberIsFinite
	case elemNumIsInteger:
		return ssa.LeafNumberIsInteger
	case elemNumIsSafeInteger:
		return ssa.LeafNumberIsSafeInteger
	}
	return ssa.LeafOther
}

// jitSearch is the answer of the search of kind, one of jitSearches, on
// recv with args (Go's GoIndexOf): the built-in's own, by its Go function,
// where nothing it does can run script or throw -- a string searched for
// primitives that are no symbol, which convert with nothing called, and
// includes's search no regular expression; an array of the realm's, dense,
// whose prototypes hold no index, which it reads with nothing asked, and
// its position such a primitive. It refuses the rest.
func (r *Runtime) jitSearch(kind int, recv Value, args []Value) (Value, bool) {
	if kind < 0 || kind >= jitSearchKinds || r.jitSearches[kind] == nil {
		return Undefined, false
	}
	if kind < jitSearchArrays {
		if !recv.IsString() {
			return Undefined, false
		}
		for _, a := range args {
			if !jitPlainOperand(a) {
				return Undefined, false
			}
		}
	} else {
		if o := r.plainArray(recv); o == nil || !r.noInheritedIndices(o) || len(args) > 1 && !jitPlainOperand(args[1]) {
			return Undefined, false
		}
	}
	fd := r.jitSearches[kind].fn()
	if fd == nil || fd.native == nil {
		return Undefined, false
	}
	v, err := fd.native(r, recv, args)
	if err != nil {
		return Undefined, false
	}
	return v, true
}

// jitParseInts is the realm's parseInt, if a plain call at the top of the
// stack calls it with one argument, a number; or nil.
func (r *Runtime) jitParseInts(sp int, in bytecode.Instr) *Object {
	if jitCallOp(in.Op) != bytecode.OpCall || in.A != 1 || r.jitParseInt == nil || !jitEncoding.Round {
		return nil
	}
	if callee := r.stack[sp-2]; !callee.IsObject() || callee.Object() != r.jitParseInt || !r.stack[sp-1].IsNumber() {
		return nil
	}
	return r.jitParseInt
}

// jitBigStrings is BigInt.prototype.toString, the realm's own, if a method
// call at the top of the stack, of no argument or one, calls it on a
// BigInt; or nil.
func (r *Runtime) jitBigStrings(sp int, in bytecode.Instr) *Object {
	argc := int(in.A)
	if jitCallOp(in.Op) != bytecode.OpCallMethod || argc > 1 || r.jitBigToString == nil {
		return nil
	}
	if callee := r.stack[sp-argc-1]; !callee.IsObject() || callee.Object() != r.jitBigToString || !r.stack[sp-argc-2].IsBigInt() {
		return nil
	}
	return r.jitBigToString
}

// jitBigString is GoBigString's string of recv, a BigInt, in the radix the
// call gives, if it gives one: BigInt.prototype.toString's, where it
// converts with nothing run and nothing thrown -- no radix, undefined, or
// a number that is an integer from 2 to 36. It refuses the rest.
func jitBigString(recv Value, args []Value) (Value, bool) {
	if !recv.IsBigInt() || bigLong(recv) {
		// A long one's digits are written by a conversion the host can
		// stop, which throws: the call's.
		return Undefined, false
	}
	radix := 10
	if len(args) > 0 && !args[0].IsUndefined() {
		if !args[0].IsNumber() {
			return Undefined, false
		}
		n := args[0].Number()
		if n != math.Trunc(n) || n < 2 || n > 36 {
			return Undefined, false
		}
		radix = int(n)
	}
	return Str(bigString(recv, radix)), true
}

// jitSubstrings is String.prototype's substring, slice or substr, the
// realm's own, if a method call at the top of the stack calls it with two
// arguments, numbers, on a string -- or charAt with one -- or nil.
func (r *Runtime) jitSubstrings(sp int, in bytecode.Instr) *Object {
	argc := int(in.A)
	if in.Op != bytecode.OpCallMethod || argc != 1 && argc != 2 {
		return nil
	}
	callee, recv := r.stack[sp-argc-1], r.stack[sp-argc-2]
	if !callee.IsObject() || !recv.IsString() || !r.jitIsSubstring(callee.Object(), argc) {
		return nil
	}
	for _, a := range r.stack[sp-argc : sp] {
		if !a.IsNumber() {
			return nil
		}
	}
	return callee.Object()
}

// jitFromCharCodes is String.fromCharCode, the realm's own, if a method
// call at the top of the stack calls it with one argument, a number, or
// nil.
func (r *Runtime) jitFromCharCodes(sp int, in bytecode.Instr) *Object {
	if in.Op != bytecode.OpCallMethod || in.A != 1 || !r.jitFromCharCode.IsObject() {
		return nil
	}
	callee := r.stack[sp-2]
	if !callee.IsObject() || callee.Object() != r.jitFromCharCode.Object() || !r.stack[sp-1].IsNumber() {
		return nil
	}
	return callee.Object()
}

// jitAllocates is the built-in a construction at the top of the stack
// makes with nothing to run, which native code makes from a pool: `new
// Array()`, the realm's own, or `new Array(n)` of a length native code
// makes (jitArrayRoom), and the room its pool's arrays are given for that
// length; or nil.
func (r *Runtime) jitAllocates(sp int, in bytecode.Instr) (*Object, int) {
	if in.Op != bytecode.OpNew || in.A > 1 {
		return nil, 0
	}
	room := 0
	if in.A == 1 {
		n := r.stack[sp-1]
		if !n.IsNumber() || n.Number() < 0 || n.Number() != math.Trunc(n.Number()) {
			return nil, 0
		}
		if room = jitArrayRoom(n.Number()); room == 0 {
			return nil, 0
		}
	}
	if c := r.stack[sp-1-int(in.A)]; c.IsObject() && c.Object() == r.proto.arrayCtor {
		return c.Object(), room
	}
	return nil, 0
}

// jitArrayRoom is the room a pooled array of `new Array(n)` is given for a
// length n (abi.ObjectPool's Room): the least of the sizes newArrayStorage
// makes an array's elements in its own allocation that n fits, as the
// existing tiers would make it; past them, the least power of two n fits
// (abi.PoolSmallRoom), its elements a slice of their own, as the existing
// tiers make them; or 0 past abi.PoolRoomLimit, the longest n the Array
// constructor makes dense, an n Go makes.
func jitArrayRoom(n float64) int {
	for _, room := range [...]int{4, 6, 8, 10, 12, abi.PoolSmallRoom} {
		if n <= float64(room) {
			return room
		}
	}
	if n > abi.PoolRoomLimit {
		return 0
	}
	return 1 << bits.Len(uint(n)-1)
}

// jitArrayFillBytes bounds the elements' storage one filling of a pool of
// arrays of more than abi.PoolSmallRoom makes (jitFillPool): a fill of
// up to abi.PoolCapacity arrays of 32 elements, one of 4,096 or more.
var jitArrayFillBytes = abi.PoolCapacity * 32 * int(valueSize)

// jitNewArrayPool is jitNewPool's for new Array(n): its arrays with room
// for room elements (abi.ObjectPool's Room), each a hole.
func (r *Runtime) jitNewArrayPool(o *Object, room int) *abi.ObjectPool {
	pool := &abi.ObjectPool{Source: unsafe.Pointer(&jitPoolSource{ctor: o}), Room: uint64(room)}
	r.jitFillPool(pool, o)
	return pool
}

// jitConstructs reports whether native code may construct with o, `new
// o(...)`, as constructWithTarget would: a plain function of this realm,
// compiled, a base constructor with no fields to give its instances, its
// prototype its own data property.
func (r *Runtime) jitConstructs(o *Object) bool {
	fd := o.fn()
	return fd != nil && fd.closure != nil && fd.native == nil && !fd.bound && fd.ctorKind == ctorBase && fd.fieldInit() == nil &&
		fd.closure.realm == r.Realm && o.class == ClassFunction && o.findOwn(atomPrototype) >= 0
}

// jitFillPool makes pool's objects for constructions with o, as
// constructWithTarget makes one -- o's prototype, its constructor's root
// shape, room for what its body adds -- while o's prototype is its own
// data property holding an object; the pool is left empty otherwise, and
// native code leaves every construction to Go.
func (r *Runtime) jitFillPool(pool *abi.ObjectPool, o *Object) {
	clear(pool.Objects[:])
	pool.Count, pool.Proto = 0, nil
	if pool.Size == 0 {
		pool.Size = abi.PoolSize
	}
	n := int(pool.Size)
	if o == r.proto.arrayCtor {
		// `new Array()`: an empty array, of the realm's prototype, with the
		// layout a property cache gives an array it meets (ensureShape),
		// which native code compares receivers' with: a method read on one
		// it made, a.push, would leave for Go to give it. `new Array(n)`'s
		// has room for pool.Room elements, holes, in its own allocation, as
		// the existing tiers make one (newArrayObject), its length 0 until
		// native code makes it n.
		room := int(pool.Room)
		if room > abi.PoolSmallRoom {
			// As many as jitArrayFillBytes holds, at least one, which is
			// what the pool counts (jitPoolBytes).
			n = max(1, min(n, jitArrayFillBytes/(room*int(valueSize))))
			pool.Size = uint64(n)
		}
		for i := range n {
			var a *Object
			if room == 0 {
				a = r.newArrayFrom(nil)
			} else {
				a = newArrayObject(r.proto.array, room)
				e := a.elemsOf()
				for j := range e {
					e[j] = elemHole
				}
				a.extOf().elems = e[:0]
			}
			r.ensureShape(a)
			pool.Objects[i] = unsafe.Pointer(a)
		}
		pool.Count, pool.Proto = uint64(n), unsafe.Pointer(r.proto.array)
		return
	}
	if proto, root, props, ok := r.jitConstructLayout(o); ok {
		jitFillObjectPool(pool, proto, root, props)
	}
}

// jitConstructLayout is what an object a construction with o makes starts
// as, as constructWithTarget makes it: o's prototype, its own data
// property, which must hold an object (ok); its constructor's root shape;
// and room for the properties its body adds. A pool's refill looks it up
// once, not for each of its objects.
func (r *Runtime) jitConstructLayout(o *Object) (proto *Object, root *shape, props int, ok bool) {
	p := o.getOwnVisible(atomPrototype)
	fd := o.fn()
	if !p.ok() || p.flags()&propAccessor != 0 || !p.value().IsObject() || fd == nil || fd.closure == nil {
		return nil, nil, 0, false
	}
	proto = p.value().Object()
	root = r.shapes.ctorRoot(fd, proto)
	if root == nil && polyable[struct{}](proto) {
		// Its object is made with proto in its table (newOrdinary), which
		// no pool's is.
		return nil, nil, 0, false
	}
	props = root.roomFor(int(fd.closure.fn.ThisProps))
	return proto, root, props, true
}

// jitFillObjectPool fills pool with objects of proto, of shape root, each
// with room for n properties, as newLiteralObject makes them: the layout
// chosen once for the refill, not for each object. Each is an allocation
// of its own, so that keeping one keeps none of the others.
func jitFillObjectPool(pool *abi.ObjectPool, proto *Object, root *shape, n int) {
	// The layout says the prototype: a constructor's or a literal's root,
	// or, where there is none, the class's for proto.
	root = root.orRoot(ClassObject, proto)
	switch n {
	case 1:
		jitFillSmallObjects[literalObject1](pool, proto, root)
	case 2:
		jitFillSmallObjects[literalObject2](pool, proto, root)
	case 0, 3:
		// None said, as for a constructor whose body adds nothing: the
		// room newLiteralObject gives it.
		jitFillSmallObjects[literalObject](pool, proto, root)
	case 4:
		jitFillSmallObjects[largeLiteralObject[[4]Value]](pool, proto, root)
	case 5:
		jitFillSmallObjects[largeLiteralObject[[5]Value]](pool, proto, root)
	case 6:
		jitFillSmallObjects[largeLiteralObject[[6]Value]](pool, proto, root)
	case 7:
		jitFillSmallObjects[largeLiteralObject[[7]Value]](pool, proto, root)
	case 8:
		jitFillSmallObjects[largeLiteralObject[[8]Value]](pool, proto, root)
	default:
		for i := range int(pool.Size) {
			pool.Objects[i] = unsafe.Pointer(newShapedLiteral(root, ClassObject, n))
		}
	}
	pool.Count, pool.Proto = pool.Size, unsafe.Pointer(proto)
}

// jitInlineObject is a pointer to one of the layouts newLiteralObject
// makes, an object with room for its properties in its own allocation:
// jitObject gives the object, its properties' room the room.
type jitInlineObject[L any] interface {
	*L
	jitObject() *Object
}

func (lo *literalObject1) jitObject() *Object {
	lo.setRoom(&lo.inline[0], len(lo.inline))
	return &lo.Object
}
func (lo *literalObject2) jitObject() *Object {
	lo.setRoom(&lo.inline[0], len(lo.inline))
	return &lo.Object
}
func (lo *literalObject) jitObject() *Object {
	lo.setRoom(&lo.inline[0], len(lo.inline))
	return &lo.Object
}
func (lo *largeLiteralObject[A]) jitObject() *Object {
	lo.setRoom(&lo.inline[0], len(lo.inline))
	return &lo.Object
}

// jitFillSmallObjects is jitFillObjectPool's for the layout L.
func jitFillSmallObjects[L any, P jitInlineObject[L]](pool *abi.ObjectPool, proto *Object, root *shape) {
	for i := range int(pool.Size) {
		o := P(new(L)).jitObject()
		o.shape = root
		o.class, o.flags = ClassObject, objExtensible|objInlineProps
		pool.Objects[i] = unsafe.Pointer(o)
	}
}

// jitNewPool is a pool of objects for constructions with o, filled.
func (r *Runtime) jitNewPool(o *Object) *abi.ObjectPool {
	pool := &abi.ObjectPool{Source: unsafe.Pointer(&jitPoolSource{ctor: o})}
	r.jitFillPool(pool, o)
	return pool
}

// jitPoolSource is what an object pool is filled from (abi.ObjectPool's
// Source): a construction's function, or a literal's root shape and how
// many fields or elements it has.
type jitPoolSource struct {
	ctor *Object
	root *shape
	n    int
	// array marks an array literal's pool; results, a call of next's pool
	// of result objects (jitFillResults); sharedPC, with sharedOf, an
	// array literal of constants whose arrays share them, made whole
	// (sharedLiteralArray) -- or, with sharedReg, register code's
	// new_array_k's, sharedPC its site's index in sharedOf's caches
	// (jitRegSharedLiteral). The four share a word.
	array     bool
	results   bool
	sharedReg bool
	sharedPC  int32
	lit       *bytecode.Constant
	// A closure's (jitFillClosures): its template, the function it is made
	// in and that function's closures' environment.
	tmpl Value
	fn   *bytecode.Function
	env  *Object
	// room is how many elements an array literal's arrays are made with
	// room for, past its own, learned of sample, the last array of the
	// pool's last filling (jitFillLiteral).
	room   int
	sample *Object
	// gen is the generator function a call's pool of generators is of
	// (jitFillGens).
	gen *Object
	// sharedOf is the closure of sharedPC's literal.
	sharedOf *closure
}

// jitRefillPool fills pool again, run out, with more objects (ssa's
// GoRefill), as an exit at its site has Go fill it, and reports whether it
// has one now: a construction's function whose prototype is no longer an
// object leaves it empty, and the site to Go.
func (r *Runtime) jitRefillPool(pool *abi.ObjectPool) bool {
	src := (*jitPoolSource)(pool.Source)
	if want := pool.Want; want != 0 && src != nil && src.ctor != nil {
		// A new Array(n) of an n past its arrays' room, or one less than
		// half of a room past abi.PoolSmallRoom: filled again, whatever it
		// holds, with arrays of n's room. One of a room past
		// abi.PoolSmallRoom, to or from, is filled with one array at first,
		// and more as it runs out, as a new pool is: a site whose lengths
		// change their room at each call is given an array a call, not a
		// pool's filling.
		pool.Want = 0
		room := jitArrayRoom(float64(want - 1))
		if room == 0 || uint64(room) == pool.Room {
			return false
		}
		if room > abi.PoolSmallRoom || pool.Room > abi.PoolSmallRoom {
			pool.Size = 1
		}
		pool.Room = uint64(room)
		r.jitFillPool(pool, src.ctor)
		return pool.Count != 0
	}
	if src == nil || pool.Count != 0 {
		return pool.Count != 0
	}
	pool.Size = min(2*max(pool.Size, abi.PoolSize), abi.PoolCapacity)
	if src.ctor != nil {
		r.jitFillPool(pool, src.ctor)
	} else if src.results {
		r.jitFillResults(pool)
	} else if src.gen != nil {
		r.jitFillGens(pool)
	} else {
		r.jitFillLiteral(&jitLiteral{root: src.root, pool: pool, n: src.n, array: src.array, lit: src.lit, tmpl: src.tmpl, fn: src.fn, env: src.env})
	}
	return pool.Count != 0
}

// jitPoolObject is an object for a construction with o, of the layout
// jitFillPool gives a pool's (jitConstructLayout), or nil if o's prototype is not its own data property holding
// an object.
func (r *Runtime) jitPoolObject(o *Object) *Object {
	proto, root, props, ok := r.jitConstructLayout(o)
	if !ok {
		return nil
	}
	return newShapedLiteral(root.orRoot(ClassObject, proto), ClassObject, props)
}

// jitRefillPools fills, at a construction that left native code at pc,
// the pool of the function it constructs with, if native code constructs
// with it there and the pool is empty, or was made for another prototype;
// it reports whether it filled an empty one.
func (r *Runtime) jitRefillPools(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) (refilled bool) {
	c := r.stack[sp-int(in.A)-1]
	if !c.IsObject() {
		return false
	}
	return r.jitRefillPoolsOf(e, pc, c.Object())
}

// jitRefillPoolsOf is jitRefillPools of the construction at pc of e's code
// -- its register pc, in register code -- with o.
func (r *Runtime) jitRefillPoolsOf(e *jitEntry, pc int, o *Object) (refilled bool) {
	// Both lists, without making one of them: this runs at every exit at
	// a construction.
	for _, x := range jitTargets(e) {
		if int(x.pc) != pc || x.obj != o || x.pool == nil {
			continue
		}
		if x.pool.Count == 0 {
			// Run out: more next time.
			x.pool.Size = min(2*max(x.pool.Size, abi.PoolSize), abi.PoolCapacity)
			r.jitFillPool(x.pool, o)
			refilled = true
		} else if x.cl == nil {
			// A built-in's, whose prototype the realm fixed.
		} else if p := o.getOwnVisible(atomPrototype); !p.ok() || !p.value().IsObject() || unsafe.Pointer(p.value().Object()) != x.pool.Proto {
			r.jitFillPool(x.pool, o)
		}
	}
	return refilled
}

// jitInliningFrequency is, in hundredths, how many calls of a callee too
// long to inline each invocation of the caller deciding its call must have
// made for it to be compiled for that caller at once (jitCalledOften):
// V8's min_inlining_frequency as V8 runs, 0.05 (ssa.MinInliningFrequency).
const jitInliningFrequency = 5

// jitLongWarming reports whether cl's function is warming up, its budget
// not spent (jitWarmFor): an entry warming, or none yet where it would be
// given one to warm, not compiled at once (jitCallThreshold).
func (r *Runtime) jitLongWarming(cl *closure) bool {
	if ce := r.jit.entryFor(cl.fn); ce != nil {
		return ce.warming
	}
	return int64(r.jitCallThreshold) > int64(max(cl.jitCalls, 1))
}

// jitCalledOften reports whether a function too long to inline (past
// ssa.MaxInline), which has no code yet (jitLongWarming), is compiled for
// the native caller deciding its call (jitState.deciding) at once rather
// than once its own budget is spent: where V8 would inline it -- with a
// feedback vector (an entry: jitFeedbackCalls invocations), no longer than
// a function inlines in all (ssa.MaxInlineTotal; V8 inlines up to 460
// bytes of its bytecode), and invoked at least jitInliningFrequency
// hundredths as often as its caller, as V8's call frequency at the site
// has it. Our inliner takes no callee past ssa.MaxInline: the compile
// stands in for it. Compiled whenever met, Typescript's long callees made
// 911 compiles a run -- 54 of them called fewer than ten times after it --
// and Mandreel's per-frame functions, called once a frame, were compiled
// at their callers' compiles, __udivmoddi4's 1,233 instructions among
// them. Waiting for their budgets, Octane's and the V8 suite's Richards
// ran HandlerTask.run, called by TaskControlBlock.run 0.37 times a call,
// and its own callees, in the tree tier for 750 calls.
func (r *Runtime) jitCalledOften(cl *closure) bool {
	return r.jitCalledOftenBy(cl, r.jit.deciding)
}

// jitCalledOftenBy is jitCalledOften for the native caller caller.
func (r *Runtime) jitCalledOftenBy(cl *closure, caller *jitEntry) bool {
	ce := r.jit.entryFor(cl.fn)
	if ce == nil || caller == nil || len(cl.fn.Code) > ssa.MaxInlineTotal {
		return false
	}
	return ce.invoked*100 >= jitInliningFrequency*max(caller.invocations(), 1)
}


// calleeCode is e, if it has code (a compile in the background installed
// at once, QJS_JIT_STRESS's background=0), or nil.
func (e *jitEntry) calleeCode() *jitEntry {
	if e.ssa != nil {
		return e
	}
	return nil
}




// setSSA gives e the code the new pipeline compiled for fn from p, which
// native callers call through its link.
func (e *jitEntry) setSSA(s *jitState, fn *bytecode.Function, p *ir.Program, code *jit.SSACode, fb *jitFeedback) {
	e.warming = false
	e.ssa, e.this, e.ssaShapes, e.ssaHolders = code, p.This, fb.shapes, fb.holders
	e.lexThis, e.loops = fb.lexThis, fb.loops
	e.setLoopSlots(fn, fb.loopSlots)
	e.ssaSerial++
	e.ssaProg = p
	e.ssaInlinedAt, e.definesUnfed, e.ssaCells, e.ssaHostsLeft, e.ssaUnresumed = fb.inlinedAt, fb.definesUnfed, fb.cells, fb.hostsLeft, fb.unresumed
	e.ssaUnentered, e.ssaKept, e.keptAt = fb.unentered, fb.kept, e.starts
	e.ssaTargets = fb.targets
	e.ssaStrings, e.ssaInlined, e.ssaPools, e.ssaMade = fb.strings, fb.inlined, fb.pools, fb.made
	e.ssaInlinedLearned = s.jitInlinedLearned(fb.inlined, e.ssaInlinedLearned[:0])
	e.ssaInlinedN = fb.inlinedN
	e.ssaVirtuals, e.ssaCalled = fb.virtuals, fb.called
	e.setHomes(s, fb.homes)
	e.ssaCallbacks, e.ssaUnmade = fb.callbacks, fb.unmade
	s.setLinks(e, fb.links)
	e.ssaKeeps = max(e.ssaKeeps, fb.keeps)
	e.link.setCode(code)
	s.callRefs = max(s.callRefs, uint64(code.CallRefs()))
	e.ssaLoop = jitLoopLength(fn)
	for _, in := range p.Code {
		e.ssaStrings = e.ssaStrings || in.Op == ir.StringMethod || in.Op == ir.StringCode
	}
	s.account(e)
}

// jitLowering is a function's programs for the new pipeline, each lowered
// once, as its kind says (jitLowerKind), and kept by the runtime: a
// compile asks for its function's and for each callee's it inlines, and a
// function is compiled again as it learns, and inlined into each of its
// callers -- lowering them again had been some 8% of a compile. A program
// is never changed once lowered. inlinable is jitInlinable's answer, once
// asked: 1 if the function may be inlined, 2 if not.
//
// bytes is what the lowering holds, its programs' (ir.Program's
// Footprint), which the JIT counts (jitState.lowerBytes); used the epoch
// it was last asked for in, as of the cache's last look (age). The cache
// drops the lowerings of functions it has no entry for to make room
// (makeRoom), the least recently used first, before any code.
type jitLowering struct {
	inlinable uint8
	done      [jitLowerKinds]bool
	progs     [jitLowerKinds]*ir.Program
	errs      [jitLowerKinds]error
	bytes     int
	used      uint64
	// hints are what its code's exits have shown it does that its
	// bytecode does not say (jitcompile.Hints): its programs are lowered
	// with them. treeAdds is how many additions its tree had marked
	// (treeSeen) when they were last taken into hints.
	hints    jitcompile.Hints
	treeAdds int32
	// serial counts its relowerings (relower): a lowering made in the
	// background from hints a later relowering replaced is stale.
	serial uint32
	// startOnly is jitcompile.StartOnly's answer for the function, once
	// asked by a call of it: 1 for no, 2 for yes.
	startOnly uint8
	// busy counts the compiles of the function's register code under way,
	// which keep its programs (regLoweringSpare).
	busy uint8
	// typedCells are the cells of its element reads and writes of numbers,
	// by PC, that learn the element types of their typed arrays
	// (zzjit_typedkind.go).
	typedCells map[int32]*Value
}

// jitLowerKind is how a program is lowered: for a function's own code
// (jitcompile.LowerSSA), for a callee inlined (LowerSSAInline), for a
// forwarding constructor inlined (LowerSSAForward), for a function with no
// loop shown to be hot (LowerSSAHot).
type jitLowerKind uint8

const (
	jitLowerSSA jitLowerKind = iota
	jitLowerInline
	jitLowerForward
	jitLowerHot
	jitLowerKinds
)

// lowering is fn's lowering, made if the runtime has none, and found used
// now.
func (s *jitState) lowering(fn *bytecode.Function) *jitLowering {
	key := weak.Make(fn)
	l := s.lowered[key]
	if l == nil {
		if s.lowered == nil {
			s.lowered = map[weak.Pointer[bytecode.Function]]*jitLowering{}
		}
		l = &jitLowering{bytes: jitLoweringFixed}
		s.lowered[key] = l
		s.lowerBytes += l.bytes
	}
	l.used = s.agedAt
	return l
}

// jitLowerWith is fn's program lowered as kind says, with hints. It reads
// only fn, which runtimes share, and hints: a worker may run it.
func jitLowerWith(fn *bytecode.Function, kind jitLowerKind, hints jitcompile.Hints) (*ir.Program, error) {
	switch kind {
	case jitLowerInline:
		return jitcompile.LowerSSAInlineWith(fn, hints)
	case jitLowerForward:
		return jitcompile.LowerSSAForwardWith(fn, hints)
	case jitLowerHot:
		return jitcompile.LowerSSAHotWith(fn, hints)
	}
	return jitcompile.LowerSSAWith(fn, hints)
}

// jitStringAddSeen notes an exit at pc, an addition in e's code that met a
// string -- the frame holding its operands as the exit wrote them -- which
// its function's programs are lowered again for (jitcompile.Hints): Go
// makes it (ir.StringAdd), and what it adds is read as it is. The code is
// compiled again at its next invocation, as for an element read learned
// (jitIndexSeen). PDF.js's font parser builds its tokens a character at a
// time, token += c, and read each character as a number, which a string
// left native code at.
func (r *Runtime) jitStringAddSeen(f *frame, e *jitEntry, pc int) {
	p := e.ssaProg
	if p == nil || pc >= len(p.Code) {
		return
	}
	in := p.Code[pc]
	if in.Op != ir.Binary || in.Operator != ir.Add {
		return
	}
	if a, b := r.jitOperand(f, p, in.Left), r.jitOperand(f, p, in.Right); !a.IsString() && !b.IsString() {
		return
	}
	l := r.jit.lowering(f.cl.fn)
	if slices.Contains(l.hints.StringAdds, int32(pc)) {
		return
	}
	before := jitCapBytes(l.hints.StringAdds)
	l.hints.StringAdds = append(l.hints.StringAdds, int32(pc))
	l.bytes += jitCapBytes(l.hints.StringAdds) - before
	r.jit.lowerBytes += jitCapBytes(l.hints.StringAdds) - before
	r.jit.relower(l)
	e.reopt = true
}

// relower drops l's programs, which are lowered again with its hints as
// they are asked for (jitLower): what was compiled from them holds them.
func (s *jitState) relower(l *jitLowering) {
	for k := range l.progs {
		if l.done[k] {
			n := l.progs[k].Footprint()
			l.bytes -= n
			s.lowerBytes -= n
		}
		l.done[k], l.progs[k], l.errs[k] = false, nil, nil
	}
	l.inlinable = 0
	l.serial++
}

// jitOperand is the value of an operand of p's instruction in f's frame as
// an exit wrote it: a literal, a local, or one on the frame's stack.
func (r *Runtime) jitOperand(f *frame, p *ir.Program, o ir.Operand) Value {
	switch {
	case o.Slot < 0:
		if o.Literal.Kind == ir.String {
			return Str(emptyString)
		}
		return Undefined
	case o.Slot < p.Locals:
		if o.Slot < len(f.locals) {
			return f.locals[o.Slot]
		}
		return Undefined
	}
	if i := f.base + o.Slot - p.Locals; i >= 0 && i < len(r.stack) {
		return r.stack[i]
	}
	return Undefined
}


// jitBigIntNote is jitBigIntSeen's learning, of either form's code: in, the
// instruction whose guard failed or which is generic, at key -- the stack
// form's site, the register form's register pc -- in e, its operands
// read by operand, treeBig and treeWide what the tree tier saw of it.
func jitBigIntNote(e *jitEntry, key int32, in ir.Instruction, operand func(ir.Operand) Value, treeBig, treeWide bool) bool {
	generic := slices.Contains(e.failed, key)
	native := false
	switch {
	case in.Op == ir.Binary && (in.Operator == ir.Add || in.Operator == ir.Sub || in.Operator == ir.Mul),
		(in.Op == ir.Binary || in.Op == ir.Branch) && (in.Operator == ir.Lt || in.Operator == ir.Le || in.Operator == ir.Gt || in.Operator == ir.Ge):
		// Generic, an addition is GoAdd's and a comparison GoCompare's,
		// whatever the site learns.
		native = !generic
		if generic && in.Operator != ir.Sub && in.Operator != ir.Mul {
			return false
		}
	case in.Op == ir.Binary && (in.Operator == ir.Div || in.Operator == ir.Mod || in.Operator == ir.Pow || in.Operator == ir.BitAnd || in.Operator == ir.BitOr ||
		in.Operator == ir.BitXor || in.Operator == ir.Shl || in.Operator == ir.Shr),
		in.Op == ir.Unary && (in.Operator == ir.Neg || in.Operator == ir.BitNot):
		native = !generic
	default:
		return false
	}
	right := in.Right
	if in.Op == ir.Unary {
		// -x and ~x: of x alone.
		right = in.Left
	}
	if in.Left.Slot < 0 || right.Slot < 0 {
		return false
	}
	a, b := operand(in.Left), operand(right)
	if !a.IsBigInt() || !b.IsBigInt() {
		return false
	}
	learned := slices.Contains(e.bigints, key)
	if learned || treeBig {
		// Its speculation, learned or the tree tier's, failed: made as one
		// of BigInts from 0 to 2^64-1 where they are, once.
		if !native || slices.Contains(e.bigwide, key) || treeWide || !jitBigUnsigned(a) || !jitBigUnsigned(b) {
			return false
		}
		e.bigwide = append(e.bigwide, key)
		e.reopt = true
		return true
	}
	e.bigints = append(e.bigints, key)
	e.reopt = true
	if !native {
		return false
	}
	if !a.isInt64Big() || !b.isInt64Big() {
		if jitBigUnsigned(a) && jitBigUnsigned(b) {
			e.bigwide = append(e.bigwide, key)
			return true
		}
		return false
	}
	x, y := a.int64Big(), b.int64Big()
	switch in.Operator {
	case ir.Neg:
		return x != math.MinInt64
	case ir.Add:
		if s := x + y; (x^s)&(y^s) < 0 {
			return false
		}
	case ir.Sub:
		if d := x - y; (x^y)&(x^d) < 0 {
			return false
		}
	case ir.Mul:
		hi, lo := bits.Mul64(uint64(x), uint64(y))
		// The signed product's high word: the unsigned one's, less each
		// operand where the other is negative.
		if x < 0 {
			hi -= uint64(y)
		}
		if y < 0 {
			hi -= uint64(x)
		}
		if hi != uint64(int64(lo)>>63) {
			return false
		}
	case ir.Div:
		if y == 0 || x == math.MinInt64 && y == -1 {
			return false
		}
	case ir.Mod:
		if y == 0 {
			return false
		}
	case ir.Pow:
		_, ok := shortArith(bytecode.OpPow, a, b)
		return ok
	}
	return true
}

// jitBigRelease gives back what of a, b and the third operand, old, the
// call of Go that made v of them may give back (ssa.CallSite's Release),
// as the tree tier's bigRelease does for its operators.
func (r *Runtime) jitBigRelease(ctx *abi.Context, v, a, b Value) {
	rel := uint8(ctx.GoOp >> abi.GoOpKindShift)
	if rel == 0 {
		return
	}
	flags := 0
	if rel&ssa.ReleaseLeft != 0 {
		flags |= treeFreshA
	}
	if rel&ssa.ReleaseRight != 0 {
		flags |= treeFreshB
	}
	old := Undefined
	if rel&ssa.ReleaseOld != 0 {
		flags |= treeOwnRes
		old = jitGoArg(ctx, 2)
	}
	r.bigRelease(v, a, b, old, flags)
}

// jitBigUnsigned reports whether v is a BigInt from 0 to 2^64-1, which is
// held in its value.
func jitBigUnsigned(v Value) bool {
	if !v.isShortBig() {
		return false
	}
	neg, _ := v.shortBigParts()
	return !neg
}

// globalSite is Global's site of a read of the global name, a write or a
// strict assignment's check, its cache site's index, or -1 for none: the
// stack form's instruction's or register code's (jitRegFeedback).
func (fb *jitFeedback) globalSite(name Atom, siteIdx int, write, check bool) (ssa.GlobalSite, bool) {
	env := fb.cl.scope()
	if env == nil {
		return ssa.GlobalSite{}, false
	}
	if !fb.r.isGlobalScope(env) {
		return fb.moduleGlobal(name, env, siteIdx, write, check)
	}
	if fb.r.lexShadows(name) {
		return fb.lexGlobal(name, env, write)
	}
	i := int32(-1)
	if siteIdx >= 0 && siteIdx < len(fb.cl.ic) {
		i = fb.cl.ic[siteIdx].idx
	}
	if i < 0 || int(i) >= int(env.valn) || env.shape.keys[i].key != name {
		i = env.findOwn(name)
	}
	if i < 0 {
		return ssa.GlobalSite{}, false
	}
	site := ssa.GlobalSite{Key: uint32(name), Index: i}
	if check {
		// A strict assignment's check: data, which the cell's guard
		// finds is there.
		if env.shape.keys[i].isAccessor() {
			return ssa.GlobalSite{}, false
		}
		return site, true
	}
	if write {
		// An assignment: a writable data property, which the code
		// checks it still is.
		if p := env.shape.keys[i]; p.isAccessor() || p.flags&propWritable == 0 {
			return ssa.GlobalSite{}, false
		}
		return site, true
	}
	if s, p := env.shape, env.shape.keys[i]; env.flags&objGlobalBindings != 0 && s.unique && !s.loose && s.tree == fb.r.shapes &&
		s.tree.globalEpoch < jitGlobalEpochs && !p.isAccessor() && p.flags&(propUninit|propPrivate|propDeleted) == 0 {
		// Plain, initialized data at its position, unshadowed: so while no
		// binding of the global object has changed but by appends, and no
		// script-level lexical binding has been declared (globalEpoch),
		// which the read checks rather than each of those.
		site.Epoch, site.Scope = uint32(s.tree.globalEpoch)+1, uintptr(unsafe.Pointer(env))
	}
	site.Typed = !env.shape.keys[i].isAccessor() && jitTypedArray(env.vals()[i])
	site.TypedKind = jitTypedKind(site.Typed, env.vals()[i])
	if p := env.shape.keys[i]; !p.isAccessor() && p.flags&(propWritable|propConfigurable) == 0 {
		switch {
		case env.vals()[i].IsUndefined():
			site.Fixed, site.Constant = true, ir.Value{Kind: ir.Undefined}
		case env.vals()[i].IsNumber():
			site.Fixed, site.Constant = true, ir.Float(env.vals()[i].Number())
		}
	}
	return site, true
}

// moduleGlobal is Global's site for name in a module's code, whose scope,
// env, is the module's environment, as V8 reads a module's variables in its
// context and an import in its exporter's cell: the bindings are fixed once
// the module is linked, none added or removed -- nothing but the module's
// own code reaches the environment -- so a site reads the object that has
// the binding, at its address (GlobalSite's Holder), with no check that
// the scope is the one it was compiled for. A binding of the environment's
// own is read there, before any of a script's lexical bindings, as the
// interpreter reads it; an import, an accessor, where the binding it names
// is (jitImportedBinding); and a name the module does not declare where the
// global object, the environment's prototype, has it, if no script-level
// lexical binding of the name shadows it. The rest is Go's.
func (fb *jitFeedback) moduleGlobal(name Atom, env *Object, siteIdx int, write, check bool) (ssa.GlobalSite, bool) {
	i := int32(-1)
	if siteIdx >= 0 && siteIdx < len(fb.cl.ic) {
		i = fb.cl.ic[siteIdx].idx
	}
	if i < 0 || int(i) >= int(env.valn) || env.shape.keys[i].key != name {
		i = env.findOwn(name)
	}
	holder := env
	shadowable := false
	switch {
	case i >= 0 && env.shape.keys[i].isAccessor():
		// An import, which only a read reads: an assignment to one throws.
		if write || check {
			return ssa.GlobalSite{}, false
		}
		h, key, ok := fb.r.jitImportedBinding(env, name)
		if !ok {
			return ssa.GlobalSite{}, false
		}
		if i = h.findOwn(key); i < 0 || h.shape.keys[i].isAccessor() {
			return ssa.GlobalSite{}, false
		}
		holder, name = h, key
	case i < 0:
		g := env.proto()
		if write || check || g == nil || fb.r.lexShadows(name) || proxyOf(g) != nil {
			return ssa.GlobalSite{}, false
		}
		if i = g.findOwn(name); i < 0 || g.shape.keys[i].isAccessor() {
			return ssa.GlobalSite{}, false
		}
		holder, shadowable = g, true
	}
	p := holder.propAt(i)
	if write && p.flags&propWritable == 0 {
		return ssa.GlobalSite{}, false
	}
	site := ssa.GlobalSite{Key: uint32(name), Index: i, Holder: uintptr(unsafe.Pointer(holder)), Shadowable: shadowable}
	site.Typed = !write && !check && jitTypedArray(p.value)
	site.TypedKind = jitTypedKind(site.Typed, p.value)
	if !write && !check && p.flags&(propWritable|propConfigurable) == 0 {
		// A const, or an import of one, once it has its value; or a
		// global that never changes, undefined, NaN or Infinity.
		switch {
		case p.value.IsUndefined() && p.flags&propUninit == 0:
			site.Fixed, site.Constant = true, ir.Value{Kind: ir.Undefined}
		case p.value.IsNumber():
			site.Fixed, site.Constant = true, ir.Float(p.value.Number())
		}
	}
	return site, true
}

// jitTypedArray reports whether v is a typed array (ssa.GlobalSite's
// Typed).
func jitTypedArray(v Value) bool {
	return v.IsObject() && v.Object().class == ClassTypedArray
}

// jitTypedKind is the element type plus one of v, a typed array if typed
// (ssa.GlobalSite's TypedKind), or 0.
func jitTypedKind(typed bool, v Value) uint8 {
	if !typed {
		return 0
	}
	if t, ok := v.Object().dataOf().(*typedArrayData); ok {
		return uint8(t.kind) + 1
	}
	return 0
}

// jitImportedBinding is where the import name binds in a module's
// environment, env, reads its value: the binding of another module's
// environment, and its name there, as linking resolved it (bindImport,
// resolveExport) -- found again here, from the modules loaded already, and
// asking the host nothing; or false where it is not one this finds, a
// namespace's export or one a star re-exports, which Go reads.
func (r *Runtime) jitImportedBinding(env *Object, name Atom) (*Object, Atom, bool) {
	var m *Module
	for _, x := range r.modules {
		if x.env == env {
			m = x
			break
		}
	}
	if m == nil {
		return nil, 0, false
	}
	local := r.atoms.name(name)
	// The binding local of m is an import: where its module exports what
	// it names, re-exported along the way or not; and that is m's own
	// binding, or another import, followed in turn. A cycle, or a chain
	// past the steps, is Go's.
	for range 16 {
		i := slices.IndexFunc(m.imports, func(imp moduleImport) bool { return imp.local == local })
		if i < 0 {
			if m.env == env {
				return nil, 0, false
			}
			return m.env, r.atoms.intern(local), true
		}
		imp := m.imports[i]
		if imp.namespace {
			return nil, 0, false
		}
		src, want := r.jitLoadedModule(m, imp.specifier), imp.imported
		if imp.isDefault {
			want = "default"
		}
		for range 16 {
			if src == nil {
				return nil, 0, false
			}
			exported, ok := src.exports[want]
			if !ok || strings.HasPrefix(exported, nsExportPrefix) {
				return nil, 0, false
			}
			spec, imported, indirect := indirectSource(exported)
			if !indirect {
				m, local = src, exported
				break
			}
			// export { x } from "...".
			src, want = r.jitLoadedModule(src, spec), imported
		}
		if src == nil || m != src {
			return nil, 0, false
		}
	}
	return nil, 0, false
}

// jitLoadedModule is the module m's request of specifier loaded, as
// resolvedNameOf finds it without asking the host, or nil.
func (r *Runtime) jitLoadedModule(m *Module, specifier string) *Module {
	plain, _ := bytecode.SplitDeferRequest(specifier)
	plain, _ = bytecode.SplitSourceRequest(plain)
	if bare, typ := bytecode.SplitModuleRequest(plain); typ != "" || bare != specifier {
		return nil
	}
	return r.requested[[2]string{m.Specifier, specifier}]
}

// lexGlobal is Global's site for a script-level lexical binding of name,
// which a let, a const or a class declared: in the realm's global scope,
// env, where the interpreter reads it (globalLexProp). The binding stays
// where it is, and a const's value, once it has one, is fixed. A write
// to a const is Go's, which throws.
func (fb *jitFeedback) lexGlobal(name Atom, env *Object, write bool) (ssa.GlobalSite, bool) {
	lex := fb.r.globalLex
	if !fb.r.isGlobalScope(env) {
		return ssa.GlobalSite{}, false
	}
	i := lex.findOwn(name)
	if i < 0 {
		return ssa.GlobalSite{}, false
	}
	p := lex.propAt(i)
	if write && p.flags&propWritable == 0 {
		return ssa.GlobalSite{}, false
	}
	site := ssa.GlobalSite{Key: uint32(name), Index: i, Lexical: true, Typed: !write && jitTypedArray(p.value)}
	site.TypedKind = jitTypedKind(site.Typed, p.value)
	if !write && p.flags&propWritable == 0 && p.value.IsNumber() {
		site.Fixed, site.Constant = true, ir.Float(p.value.Number())
	}
	return site, true
}

// stringPropertyOf is StringProperty's, of either form's code: the method
// read at pc, its cache c, of key; big says another tier saw it meet a
// BigInt.
func (fb *jitFeedback) stringPropertyOf(pc int, c *propCache, key Atom, big bool) (ssa.PropertySite, bool) {
	e := fb.e
	if c.shape != nil && c.shape != noShape || slices.ContainsFunc(e.poly, func(p jitPolySite) bool { return int(p.pc) == pc }) {
		return ssa.PropertySite{}, false
	}
	proto, kind := fb.r.proto.str, ssa.PrimitiveString
	str := slices.Contains(e.strSites, uint32(pc))
	switch {
	case str:
	case slices.Contains(e.bigSites, uint32(pc)) || big:
		// A read that met BigInt receivers and no string: off
		// BigInt.prototype, which the site names (Holders).
		proto, kind = fb.r.proto.bigint, ssa.PrimitiveBigInt
	case slices.Contains(e.numSites, uint32(pc)):
		// Numbers: off Number.prototype so.
		proto, kind = fb.r.proto.number, ssa.PrimitiveNumber
	case slices.Contains(e.boolSites, uint32(pc)):
		proto, kind = fb.r.proto.boolean, ssa.PrimitiveBoolean
	}
	if key.IsIndex() || key == atomLength || proto == nil {
		return ssa.PropertySite{}, false
	}
	if kind == ssa.PrimitiveString && !str &&
		(c.fills != 0 || fb.r.proto.object == nil || fb.r.proto.object.findOwn(key) >= 0) {
		return ssa.PropertySite{}, false
	}
	sh := fb.r.ensureProtoShape(proto)
	i := proto.findOwn(key)
	if sh == nil || sh == noShape || i < 0 || proto.shape.keys[i].flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 {
		return ssa.PropertySite{}, false
	}
	k := fb.keep()
	k.shapes = append(k.shapes, remember(sh))
	site := ssa.PropertySite{Key: uint32(key), Shape: uintptr(unsafe.Pointer(sh)), Index: i, Primitive: kind}
	if kind != ssa.PrimitiveString {
		k.holders = append(k.holders, proto)
		site.Holders[0] = ssa.Holder{Object: uintptr(unsafe.Pointer(proto)), Shape: site.Shape}
	}
	return site, true
}

// writeCases are what the write at pc met at objects of shapes other than
// site's (jitPolyWriteSeen): an own property it wrote, or a transition it
// added the property along (fb.add), which the code tries in turn.
func (fb *jitFeedback) writeCases(pc int, site ssa.PropertySite) ([]ssa.PropertyCase, []*ssa.PropertyAdd) {
	var cases []ssa.PropertyCase
	var adds []*ssa.PropertyAdd
	for _, p := range fb.e.poly {
		if int(p.pc) != pc {
			continue
		}
		for i := range p.cases {
			c := &p.cases[i]
			from := uintptr(unsafe.Pointer(c.shape))
			if from == site.Shape || site.Add != nil && from == site.Add.From {
				continue
			}
			if c.next != nil {
				if a := fb.add(c); a != nil {
					adds = append(adds, a)
				}
				continue
			}
			if c.p1 == nil && c.idx >= 0 && c.idx < c.shape.n {
				k := fb.keep()
				k.shapes = append(k.shapes, remember(c.shape))
				cases = append(cases, ssa.PropertyCase{Shape: from, Index: c.idx})
			}
		}
		for _, d := range p.adds {
			if from := uintptr(unsafe.Pointer(d.from)); from == site.Shape || site.Add != nil && from == site.Add.From {
				continue
			}
			if a := fb.deepAdd(d); a != nil {
				adds = append(adds, a)
			}
		}
	}
	return cases, adds
}

// deepAdd is d, an addition along a prototype chain deeper than a cache
// holds, as ssa.PropertyAdd has it, the code keeping its shapes and
// prototypes; or nil where its next shape wants its table's index built,
// as fb.add leaves to Go.
func (fb *jitFeedback) deepAdd(d jitDeepAdd) *ssa.PropertyAdd {
	if d.next.index == nil && int(d.next.n) > linearScanLimit {
		return nil
	}
	k := fb.keep()
	k.shapes = append(k.shapes, remember(d.from), remember(d.next))
	a := &ssa.PropertyAdd{From: uintptr(unsafe.Pointer(d.from)), Next: uintptr(unsafe.Pointer(d.next)), Flags: uint8(d.next.flags)}
	for j, up := range d.up {
		h := ssa.Holder{Object: uintptr(unsafe.Pointer(up)), Shape: uintptr(unsafe.Pointer(d.shapes[j]))}
		k.shapes, k.holders = append(k.shapes, remember(d.shapes[j])), append(k.holders, up)
		if j < len(a.Protos) {
			a.Protos[j] = h
		} else {
			a.More = append(a.More, h)
		}
	}
	return a
}

// seenCases are the cases of the property site at pc, as its cache met
// them before the shape it holds now, compiled for site (propCache.seen):
// a read's found on the receiver or on a prototype a cache holds, a
// write's own properties; up to jitPropertyCases. An inlined callee's
// site is compiled for them, as V8 compiles a site inlined for the maps
// its inline cache met: at any other shape it deoptimizes. A shape the one
// the cache holds came from by adding properties takes what room the
// others leave, as a root's does (jitAddReadCase).
func (fb *jitFeedback) seenCases(pc int, site ssa.PropertySite) []ssa.PropertyCase {
	in := fb.fn.Code[pc]
	if int(in.B) >= len(fb.cl.ic) || fb.cl.ic[in.B].seenOf() == nil || site.Holders[0].Object != 0 && in.Op == bytecode.OpSetProp {
		return nil
	}
	c := &fb.cl.ic[in.B]
	write := in.Op == bytecode.OpSetProp
	var cases []ssa.PropertyCase
	k := fb.keep()
	// The entries the cache holds shapes of besides its own, then those of
	// the ones superseded.
	seen := c.seenOf()
	entries := seen.entries[:seen.n]
	for i := range 2 * len(entries) {
		x := entries[i%len(entries)]
		if len(cases)+1 >= jitPropertyCases {
			break
		}
		from := uintptr(unsafe.Pointer(x.shape))
		if x.shape == nil || x.shape == noShape || x.getter || x.idx < 0 || x.next != nil || from == site.Shape ||
			jitSuperseded(x.shape, c.shape) != (i >= len(entries)) || write && x.p1 != nil {
			continue
		}
		if slices.ContainsFunc(cases, func(p ssa.PropertyCase) bool { return p.Shape == from }) {
			continue
		}
		pcase := ssa.PropertyCase{Shape: from, Index: x.idx}
		k.shapes = append(k.shapes, remember(x.shape))
		for i, h := range [2]struct {
			p *Object
			s *shape
		}{{x.p1, x.s1}, {x.p2, x.s2}} {
			if h.p == nil {
				break
			}
			pcase.Holders[i] = ssa.Holder{Object: uintptr(unsafe.Pointer(h.p)), Shape: uintptr(unsafe.Pointer(h.s))}
			k.shapes, k.holders = append(k.shapes, remember(h.s)), append(k.holders, h.p)
		}
		cases = append(cases, pcase)
	}
	return cases
}

// seenAdds are the additions the property write at pc, an inlined
// callee's, made as its cache met them besides the one site is compiled for
// (propCache.seen, poly) -- and those along prototype chains deeper than it
// holds (propSeen's deep) -- up to jitPropertyCases: an initializer's that a
// class's subclasses share, as Box2D's static ones are, each class's
// objects of a shape of their own, whose additions had each been an exit.
func (fb *jitFeedback) seenAdds(pc int, site ssa.PropertySite) []*ssa.PropertyAdd {
	in := fb.fn.Code[pc]
	if int(in.B) >= len(fb.cl.ic) {
		return nil
	}
	c := &fb.cl.ic[in.B]
	var adds []*ssa.PropertyAdd
	note := func(x *propCache) {
		if len(adds)+1 >= jitPropertyCases || x.next == nil || x.shape == nil || x.shape == noShape || x.getter {
			return
		}
		from := uintptr(unsafe.Pointer(x.shape))
		if from == site.Shape || site.Add != nil && from == site.Add.From ||
			slices.ContainsFunc(adds, func(a *ssa.PropertyAdd) bool { return a.From == from }) {
			return
		}
		if a := fb.add(x); a != nil {
			adds = append(adds, a)
		}
	}
	if seen := c.seenOf(); seen != nil {
		for i := range seen.n {
			note(&seen.entries[i])
		}
	}
	if c.poly != nil {
		for i := range c.poly.n {
			note(&c.poly.entries[i])
		}
	}
	if seen := c.seenOf(); seen != nil && int(in.A) < len(fb.cl.names) {
		// The additions along chains deeper than a cache holds -- a
		// subclass's object's, two prototypes below the class whose
		// initializer adds the property -- as the chain is now.
		for _, d := range seen.deep {
			from := uintptr(unsafe.Pointer(d.from))
			if len(adds)+1 >= jitPropertyCases || from == site.Shape || site.Add != nil && from == site.Add.From ||
				slices.ContainsFunc(adds, func(a *ssa.PropertyAdd) bool { return a.From == from }) {
				continue
			}
			if da, ok := fb.r.jitDeepAddFrom(d.from, d.next, d.proto, fb.cl.names[in.A]); ok {
				if a := fb.deepAdd(da); a != nil {
					adds = append(adds, a)
				}
			}
		}
	}
	return adds
}

// cases are the shapes other than first the read at pc met (jitPolySeen),
// as ssa.PropertyCase has them: the code keeps them and their holders.
func (fb *jitFeedback) cases(pc int, first uintptr) []ssa.PropertyCase {
	var cases []ssa.PropertyCase
	for _, p := range fb.e.poly {
		if int(p.pc) != pc {
			continue
		}
		for _, c := range p.cases {
			if uintptr(unsafe.Pointer(c.shape)) == first {
				continue
			}
			k := ssa.PropertyCase{Shape: uintptr(unsafe.Pointer(c.shape)), Index: c.idx}
			fb.shapes = append(fb.shapes, c.shape)
			for i, h := range [2]struct {
				p *Object
				s *shape
			}{{c.p1, c.s1}, {c.p2, c.s2}} {
				if h.p == nil {
					break
				}
				k.Holders[i] = ssa.Holder{Object: uintptr(unsafe.Pointer(h.p)), Shape: uintptr(unsafe.Pointer(h.s))}
				fb.shapes, fb.holders = append(fb.shapes, h.s), append(fb.holders, h.p)
			}
			cases = append(cases, k)
		}
		for i := range p.reads {
			if d := &p.reads[i]; uintptr(unsafe.Pointer(d.shape)) != first {
				cases = append(cases, fb.deepCase(d))
			}
		}
	}
	return cases
}

// deepCase is a read's case past what a cache holds (jitDeepRead), whose
// shapes and prototypes the code keeps.
func (fb *jitFeedback) deepCase(d *jitDeepRead) ssa.PropertyCase {
	k := ssa.PropertyCase{Shape: uintptr(unsafe.Pointer(d.shape)), Index: d.idx}
	keep := fb.keep()
	keep.shapes = append(keep.shapes, d.shape)
	for i, up := range d.up {
		h := ssa.Holder{Object: uintptr(unsafe.Pointer(up)), Shape: uintptr(unsafe.Pointer(d.shapes[i]))}
		keep.shapes, keep.holders = append(keep.shapes, d.shapes[i]), append(keep.holders, up)
		if i < len(k.Holders) {
			k.Holders[i] = h
		} else {
			k.More = append(k.More, h)
		}
	}
	return k
}

// deepCases are the cases of the read at pc, whose cache knew nothing --
// none it could hold, the property found past its two prototypes -- as
// e's exits learned them (cases), and as the cache noted them
// (jitNoteDeepRead), for an inlined callee's site too: up to
// jitPropertyCases.
func (fb *jitFeedback) deepCases(pc int) []ssa.PropertyCase {
	var c *propCache
	if in := fb.fn.Code[pc]; int(in.B) < len(fb.cl.ic) {
		c = &fb.cl.ic[in.B]
	}
	return fb.deepCasesOf(pc, c)
}

// deepCasesOf is deepCases', of either form's code: the read at pc, its
// cache c (nil for none).
func (fb *jitFeedback) deepCasesOf(pc int, c *propCache) []ssa.PropertyCase {
	var cases []ssa.PropertyCase
	if fb.e != nil && fb.root == nil {
		cases = fb.cases(pc, 0)
	}
	if c != nil && c.seenOf() != nil {
		reads := c.seenOf().reads
		for i := range reads {
			d := &reads[i]
			if len(cases) < jitPropertyCases && !slices.ContainsFunc(cases, func(k ssa.PropertyCase) bool { return k.Shape == uintptr(unsafe.Pointer(d.shape)) }) {
				cases = append(cases, fb.deepCase(d))
			}
		}
	}
	return cases
}



// jitKnownReceiver is the object the property read at q of cl's code
// reads from, as the instruction just before it gives it: the object a
// captured binding holds now, or a RegExp a regular expression literal
// makes, one like those it makes -- a guess, which the read's checks test.
func (r *Runtime) jitKnownReceiver(cl *closure, p *ir.Program, q int) (*Object, bool) {
	fn := cl.fn
	if q <= 0 || q >= len(fn.Code) || p.Maps[q].Depth < 0 || p.Maps[q-1].Depth != p.Maps[q].Depth-1 {
		return nil, false
	}
	var v Value
	switch prev := fn.Code[q-1]; prev.Op {
	case bytecode.OpGetUpvalue, bytecode.OpGetUpvalueCheck:
		if int(prev.A) >= len(cl.upvalues) || cl.upvalues[prev.A] == nil {
			return nil, false
		}
		v = cl.upvalues[prev.A].get()
	case bytecode.OpNewRegExp:
		if int(prev.A) >= len(fn.Constants) || fn.Constants[prev.A].Kind != bytecode.ConstRegExp {
			return nil, false
		}
		var err error
		if v, err = r.newRegExpLiteral(&fn.Constants[prev.A]); err != nil {
			return nil, false
		}
	default:
		return nil, false
	}
	if !v.IsObject() {
		return nil, false
	}
	return v.Object(), true
}


// siteFrom is the site of a property instruction, in, as cache c knows it.
func (fb *jitFeedback) siteFrom(in bytecode.Instr, c *propCache) (ssa.PropertySite, bool) {
	return fb.siteOf(fb.cl.names[in.A], in.Op == bytecode.OpSetProp, c)
}

// siteOf is the site of a read of key, or a write, as cache c knows it.
func (fb *jitFeedback) siteOf(key Atom, write bool, c *propCache) (ssa.PropertySite, bool) {
	site := ssa.PropertySite{Key: uint32(key)}
	k := fb.keep()
	if write && c.next != nil && c.next != setterNext && !c.getter {
		site.Add = fb.add(c)
		return site, true
	}
	if c.shape == nil || c.shape == noShape || c.shape == dictShape || c.cell != nil || c.getter || c.next != nil || c.idx < 0 {
		// Nothing native code is compiled for: an entry for dictionary
		// objects (dictShape) answers by a lookup in the object's table,
		// and one of a chain's cell (protoCell) by the cell.
		return site, true
	}
	if c.p1 == nil {
		if c.idx < c.shape.n {
			k.shapes = append(k.shapes, remember(c.shape))
			site.Shape, site.Index = uintptr(unsafe.Pointer(c.shape)), c.idx
		}
		return site, true
	}
	// Found on a prototype (propCache.holder): the receiver's shape, then
	// each prototype's, which the code compares as the cache does, and
	// the index in the last one's table. A read only: a write makes a
	// property of the receiver's own.
	last := c.s1
	if c.p2 != nil {
		last = c.s2
	}
	if write || c.s1 == nil || c.s1 == noShape || c.p2 != nil && (c.s2 == nil || c.s2 == noShape) || c.idx >= last.n {
		return site, true
	}
	k.shapes = append(k.shapes, remember(c.shape), remember(c.s1))
	k.holders = append(k.holders, c.p1)
	site.Holders[0] = ssa.Holder{Object: uintptr(unsafe.Pointer(c.p1)), Shape: uintptr(unsafe.Pointer(c.s1))}
	if c.p2 != nil {
		k.shapes = append(k.shapes, remember(c.s2))
		k.holders = append(k.holders, c.p2)
		site.Holders[1] = ssa.Holder{Object: uintptr(unsafe.Pointer(c.p2)), Shape: uintptr(unsafe.Pointer(c.s2))}
	}
	site.Shape, site.Index = uintptr(unsafe.Pointer(c.shape)), c.idx
	return site, true
}

// add is what a write whose cache adds its property adds (propCache.adds),
// as ssa.PropertyAdd has it, or nil for one native code leaves to Go: one
// whose next shape wants its table's index built (appendTransition). The
// code keeps the shapes and prototypes it compares.
func (fb *jitFeedback) add(c *propCache) *ssa.PropertyAdd {
	next := c.next
	if c.shape == nil || c.shape == noShape || next.index == nil && int(next.n) > linearScanLimit {
		return nil
	}
	if c.cell != nil {
		// An addition past a chain longer than the entry holds, which it
		// checks by the chain's cell (fillChainStore): compiled as the
		// additions along such chains are, for each prototype as it is.
		if !c.cell.valid {
			return nil
		}
		if d, ok := fb.r.jitDeepAddFrom(c.shape, next, c.p1, next.key); ok {
			return fb.deepAdd(d)
		}
		return nil
	}
	add := &ssa.PropertyAdd{From: uintptr(unsafe.Pointer(c.shape)), Next: uintptr(unsafe.Pointer(next)), Flags: uint8(next.flags)}
	for i, h := range [2]struct {
		p *Object
		s *shape
	}{{c.p1, c.s1}, {c.p2, c.s2}} {
		if h.p == nil {
			break
		}
		if h.s == nil || h.s == noShape {
			return nil
		}
		add.Protos[i] = ssa.Holder{Object: uintptr(unsafe.Pointer(h.p)), Shape: uintptr(unsafe.Pointer(h.s))}
	}
	k := fb.keep()
	k.shapes = append(k.shapes, remember(c.shape), remember(next))
	for i, h := range [2]struct {
		p *Object
		s *shape
	}{{c.p1, c.s1}, {c.p2, c.s2}} {
		if add.Protos[i].Object != 0 {
			k.shapes, k.holders = append(k.shapes, remember(h.s)), append(k.holders, h.p)
		}
	}
	return add
}

// jitSlot is a frame's slot as the JIT numbers them: its locals, its
// captured bindings, the receiver if the code reads it (this, its
// program's This: from the context, where Go put it), then its operands.
func (r *Runtime) jitSlot(f *frame, this bool, i int) *Value {
	n, u := f.cl.fn.LocalCount, len(f.cl.upvalues)
	switch {
	case i < n:
		return &f.locals[i]
	case i < n+u:
		return f.cl.upvalues[i-n].slot
	case this && i == n+u:
		return (*Value)(unsafe.Pointer(&r.jit.ssaCtx.This))
	}
	if this {
		u++
	}
	return &r.stack[f.base+i-n-u]
}

// jitStringMethod is String.prototype.charCodeAt for native code: the
// intrinsic, while the prototype still has it as it was made; and
// otherwise nothing.
func (r *Runtime) jitStringMethod() Value {
	if r.jitCharCodeAt.ref == nil {
		return Value{}
	}
	s := r.jit
	if !s.charCodeAtKnown {
		s.charCodeAtKey, s.charCodeAtKnown = r.atoms.intern("charCodeAt"), true
	}
	m := r.proto.str.getOwn(s.charCodeAtKey)
	if !m.ok() || m.flags()&^propDefault != 0 || !m.value().StrictEquals(r.jitCharCodeAt) {
		return Value{}
	}
	return r.jitCharCodeAt
}

// jitSource is the value a run-time source names (internal/jit/ssa,
// origin.go), which word holds as native code wrote it: the value's
// address -- a slot's in the frame, the receiver's in a context, a captured
// binding's, or a heap cell's native code loaded a reference from -- or nil
// for a primitive's 0. It still holds the reference: native code stores no
// pointer, and nothing else has run since it read it. Reading the address
// back as a pointer is outside unsafe.Pointer's documented rules, and sound
// while Go's heap does not move (internal/jit/README.md, "Reading the heap
// in place").
func jitSource(word *uint64) *Value {
	if *word == 0 {
		return nil
	}
	return *(**Value)(unsafe.Pointer(word))
}

// jitApplyRecords writes the slots records name (abi.Record): the
// references moved, and the primitives put where a reference was,
// numbered as the code that left has them (this, jitSlot's). They are a
// native call's caller's state, which the call described rather than
// wrote, made records of (jit.ExpandCall) when its callee left native
// code. An exit applies its own records natively as it leaves (mir's
// recordsTail), whether the collector marks or not, or leaves its frame
// to its description (jit.ApplyExit): at an exit none are left. The slots
// records read hold their values from entry until the first write, so
// every one is read first; one record, the usual case, reads its value and
// writes it.
func (r *Runtime) jitApplyRecords(f *frame, this bool, ctx *abi.Context) {
	n := int(ctx.Records)
	if n == 0 {
		return
	}
	r.jit.ssaRecords += uint64(n)
	const flags = abi.RecordScalar | abi.RecordMaybe | abi.RecordDirect
	if n == 1 {
		rec := &ctx.Record[0]
		*r.jitSlot(f, this, int(rec.Slot&^flags)) = r.jitRecordValue(f, this, ctx, 0)
		return
	}
	var buf [8]Value
	src := buf[:0]
	for i := range ctx.Record[:n] {
		src = append(src, r.jitRecordValue(f, this, ctx, i))
	}
	for i, rec := range ctx.Record[:n] {
		*r.jitSlot(f, this, int(rec.Slot&^flags)) = src[i]
	}
}

// jitLevelSlot is a slot of a level's frame and the word it held before the
// level's records wrote over it (jitKeepSlots).
type jitLevelSlot struct {
	at  *Value
	old Value
}

// jitKeepSlots keeps what the records of c, l's context, are to write over
// in f, l's frame, for its code to find if it goes on after its call
// (jitRestoreSlots). Native code writes a frame only at an exit, and a value
// it took from a slot goes on reading its pointer word there, whatever the
// local has been assigned since: the slot is that value's until it dies, and
// may be all that keeps its object alive, which what is kept here does in
// its place. Records give the slot the local's value, for what reads the
// frame while Go finishes the call. A local of the function's closures, in
// a cell the frame holds open, code does not read in the slot, and what the
// callee assigned it stays assigned; a captured binding's cell is no slot of
// the frame's.
func (r *Runtime) jitKeepSlots(l *jitNativeLevel, f *frame, this bool, c *abi.Context) {
	s := r.jit
	l.slotsFrom = len(s.levelSlots)
	const flags = abi.RecordScalar | abi.RecordMaybe | abi.RecordDirect
	fn := f.cl.fn
	for _, rec := range c.Record[:c.Records] {
		slot := int(rec.Slot &^ flags)
		if slot < len(fn.Locals) && fn.Locals[slot].Captured || slot >= fn.LocalCount && slot < fn.LocalCount+len(f.cl.upvalues) {
			continue
		}
		at := r.jitSlot(f, this, slot)
		s.levelSlots = append(s.levelSlots, jitLevelSlot{at, *at})
	}
	l.slotsTo = len(s.levelSlots)
}

// jitRestoreSlots gives l's frame back what its code left there
// (jitKeepSlots), last first, as l goes on natively after its call.
func (r *Runtime) jitRestoreSlots(l *jitNativeLevel) {
	slots := r.jit.levelSlots[l.slotsFrom:l.slotsTo]
	for i := len(slots) - 1; i >= 0; i-- {
		*slots[i].at = slots[i].old
	}
}

// jitRecordValue is the value a record writes, read from the frame as the
// exit left it.
func (r *Runtime) jitRecordValue(f *frame, this bool, ctx *abi.Context, i int) Value {
	switch rec := &ctx.Record[i]; {
	case rec.Slot&abi.RecordDirect != 0:
		slot := abi.Slot{Num: rec.Word, Ref: ctx.RecordRef[i]}
		return *(*Value)(unsafe.Pointer(&slot))
	case rec.Slot&abi.RecordScalar != 0:
		return Value{num: math.Float64frombits(rec.Word)}
	case rec.Slot&abi.RecordMaybe != 0:
		if s := jitSource(&rec.Arg); s != nil && s.ref != nil {
			return *s
		}
		return Value{num: math.Float64frombits(rec.Word)}
	default:
		return *r.jitSlot(f, this, int(rec.Arg))
	}
}

// jitSSAStats counts what one function's new-pipeline code does. Each
// entry starts a native stretch, which ends in a return, an exit to Go or
// a failed guard (ended counts those two), or a poll; work estimates, in
// slot IR instructions, what the stretches did natively.
type jitSSAStats struct {
	entries, hosts, guards, polls, records uint64
	ended, work                            uint64
	// goCalls counts the calls native code had Go make (jitGoCall).
	goCalls uint64
	// calleeGoCalls counts those it made as a native callee
	// (jitGoCallNested), and calleeWork the work it did natively before
	// each, as jitSSAProfit estimates a stretch's (jitCalleeProfit), since
	// its calls in (jitLink's in) were last counted from 0.
	calleeGoCalls, calleeWork uint64
	// runGoCalls counts those it made in runs begun at a loop that Go
	// makes for good, and runWork the work it did natively before each
	// (jitRunGoProfit); osrStretches the
	// stretches begun at a loop by on-stack replacement, and osrWork the
	// work they did (jitSSAProfit) -- each since Go last began entering the
	// code again (jitRetrySlow).
	runGoCalls, runWork   uint64
	osrStretches, osrWork uint64
	// judged is how many stretches, calls of Go included, there had been
	// when jitSSAProfit last judged the code.
	judged uint64
	// workBase and stretchBase are work and the stretches when Go last
	// began entering the code again (jitRetrySlow): it is judged by what
	// it has done since.
	workBase, stretchBase uint64
	// guardsAt counts failed guards by their site (abi.Context.ExitSite:
	// the slot IR PC of the operation, or -1 for an entry's speculation);
	// nil until one fails.
	guardsAt map[int32]uint64
}

// jitSSAProbe is how many native stretches a function runs between looks
// at whether its native code pays (jitSSAProfit), and jitSSAMinWork the
// work a stretch must do, on average, for it to: an entry, and the exit or
// return that ends the stretch, cost what the tree tier takes for about
// that many instructions (the V8 suite's medians of five: 10 left Richards'
// task methods, called from the tree tier and calling out again, native at
// a loss; 20 is best overall, 1,374.8 ms against 1,413.6).
const (
	jitSSAProbe   = 64
	jitSSAMinWork = 20
)

// jitSSAProfit accounts for a native stretch that started at pc, with the
// back-edge counter at edges, and ended as ctx says; every jitSSAProbe
// stretches it decides whether the function's native code pays. Code whose
// stretches do little work before they end -- leaving for Go, a method
// call at every iteration, or returning, a small method the tree tier calls
// that calls out again -- costs more there than it saves, and the tree tier
// runs it from then on. The work is estimated: the distance from the entry
// to the exit or return (the return's block), in slot IR instructions, and
// a loop's mean length for each back-edge taken natively.
//
// The stretches of a run begun at a loop by on-stack replacement (osr)
// are judged apart besides. Code judged slow is entered so at a loop that
// has run a full budget in the existing tiers (tryJITAt), the judgement
// being of Go's entries at the start, whose stretches are a short call's.
// But where the run's stretches do too little too -- the loop leaves at
// each turn, for a using declaration's resources, say -- each entry ran a
// turn natively and left the frame for the interpreter until the next
// budget, a loop run in the bytecode interpreter: Go does not enter such
// code at a loop either (loopSlow), and the tree runs the loop.
func (r *Runtime) jitSSAProfit(e *jitEntry, f *frame, ctx *abi.Context, start, edges int, osr bool) {
	st := &e.ssaStats
	work := uint64(0)
	if back := edges - (r.backEdges - r.jit.pollCredit); back > 0 {
		work = uint64(back) * uint64(e.ssaLoop)
	}
	if ctx.ExitKind == abi.ExitHost || ctx.ExitKind == abi.ExitDeopt || ctx.ExitKind == abi.ExitReturn {
		if ctx.ExitKind != abi.ExitReturn {
			st.ended++
		}
		if site := int64(ctx.ExitSite); site > int64(start) {
			work += uint64(site - int64(start))
		}
	}
	st.work += work
	if osr {
		st.osrStretches++
		st.osrWork += work
		if st.osrStretches%jitSSAProbe == 0 && st.osrWork < st.osrStretches*jitSSAMinWork {
			e.entrySlow, e.loopSlow = true, true
		}
	}
	// A call of Go a stretch makes costs it what an exit and an entry
	// would, which it saves (ssa.GoCall): each counts as a stretch, so a
	// method whose call Go now makes is judged as it was when it left --
	// and as often: every jitSSAProbe stretches, its calls of Go among
	// them. Judged every jitSSAProbe entries, a function entered a few
	// times, each a loop of a hundred thousand calls of built-ins --
	// RegExp's run blocks -- was never judged.
	stretches := st.entries + st.goCalls
	if stretches-st.judged < jitSSAProbe {
		return
	}
	st.judged = stretches
	if st.work-st.workBase < (stretches-st.stretchBase)*jitSSAMinWork {
		// Go stops entering it; native callers, which pay neither, go on
		// calling it (NativeCalls).
		e.entrySlow = true
	}
}

// jitRetrySlow counts a call the existing tiers made of code Go does not
// enter (entrySlow), and has Go enter it again after jitNativeRetry calls,
// doubled each time it was found not to pay, up to jitNativeBackoffs times,
// judging it by what it does from then on (jitSSAProfit): code judged on its
// first stretches -- its calls not yet decided, made by Go; its sites
// without feedback -- may pay once it has learned, as a native callee does
// (jitRetryNative). Judged once, for good, AWFY's Richards' runTask had
// every call made by Go after its one call's target was learned. Code
// compiled for native callers alone (ssaCallee) Go does not enter.
func jitRetrySlow(e *jitEntry) {
	if e.ssaCallee || e.ssa == nil || e.slowCalls-e.slowSince < jitNativeRetry<<e.slowBackoff {
		return
	}
	st := &e.ssaStats
	e.entrySlow, e.loopSlow = false, false
	e.slowSince, e.slowBackoff = e.slowCalls, min(e.slowBackoff+1, jitNativeBackoffs)
	st.judged = st.entries + st.goCalls
	st.workBase, st.stretchBase = st.work, st.judged
	st.runGoCalls, st.runWork, st.osrStretches, st.osrWork = 0, 0, 0, 0
}

// jitLoopLength is the mean length of fn's loops, in instructions: the
// distance each backward branch jumps; 1 for a function with none.
func jitLoopLength(fn *bytecode.Function) uint32 {
	total, loops := 0, 0
	for pc, in := range fn.Code {
		switch in.Op {
		case bytecode.OpJump, bytecode.OpJumpIfFalse, bytecode.OpJumpIfTrue, bytecode.OpJumpIfCmpFalse:
			if int(in.A) <= pc {
				total += pc - int(in.A) + 1
				loops++
			}
		}
	}
	if loops == 0 {
		return 1
	}
	return uint32(total / loops)
}

// errGoOnNatively ends the interpreter's run of a frame Go goes on in
// natively, at a loop's header (jitGoOn).
var errGoOnNatively = errors.New("jit: go on natively")

// jitRunDone ends a run runSSAIn began in context idx, ctxTop top before
// it. What native code kept (abi.Context.Keep), and the pointer words its
// native calls recorded (RecordRef), its callees' too, are not kept past
// it; nor its receiver and captured bindings, nor what its native calls'
// callees ran with -- closure, receiver, result, keep cells -- in the
// contexts past it, each a frame gone once it returned, as V8's are: one
// receiver left there kept a benchmark's whole object graph alive after it
// ended. The contexts the calls ran in go from the one past it, each given
// a closure by the call, up to the first without, which no call ran in
// since Go last cleared it here (a run of Go's there gives it none, and
// clears those past it itself): none past that is looked at.
func (r *Runtime) jitRunDone(e *jitEntry, idx, top int) {
	s := r.jit
	e.ssaRuns--
	s.runEntries[len(s.runEntries)-1] = nil
	s.runEntries = s.runEntries[:len(s.runEntries)-1]
	s.ctxTop = top
	if idx < len(s.goRuns) {
		s.goRuns[idx] = jitGoRun{}
	}
	if e.ssaHomes != e.ssaHomesNow {
		e.setHomes(s, e.ssaHomesNow)
	}
	c := s.ssaCtxs[idx]
	clear(c.Keep[:e.ssaKeeps])
	if e.osrOf != nil {
		s.osrRunDone(e)
	}
	c.This, c.Upvalues, c.Instance = abi.Slot{}, nil, abi.Slot{}
	// The frames made for the native levels of this run's calls of Go
	// (jitLevels), which only its calls share: their closures and
	// receivers had kept a benchmark's graph alive once it ended.
	if lv := &s.levels; len(lv.at) != 0 && idx <= lv.root {
		clear(lv.at)
		lv.at, lv.serial, lv.n = lv.at[:0], 0, 0
	}
	// Past RecordHigh, the pointer words native calls wrote themselves,
	// which Go did not expand (abi.CallRecord's Ref).
	if n := max(c.RecordHigh, s.callRefs); n != 0 {
		clear(c.RecordRef[:n])
		c.RecordHigh = 0
	}
	for i := idx + 1; i < len(s.ssaCtxs); i++ {
		c := s.ssaCtxs[i]
		if c.Closure == nil {
			break
		}
		if n := max(c.RecordHigh, s.callRefs); n != 0 {
			clear(c.RecordRef[:n])
			c.RecordHigh = 0
		}
		clear(c.Keep[:s.ssaKeeps])
		c.Closure, c.Upvalues, c.Cells = nil, nil, nil
		c.This, c.RetValue, c.Instance = abi.Slot{}, abi.Slot{}, abi.Slot{}
	}
}

// jitShared is what every context native code runs in shares with the one
// it was called from (abi.Context): where the global names are, the
// VM's stack. How deep calls may go they share in one place
// (jitState.bounds).
type jitShared struct {
	global, lexNames, globalLex, strMethods, stackBase unsafe.Pointer
	stackEnd                                           uint64
	// charCodeAt is String.prototype.charCodeAt as native code calls it
	// (jitStringMethod), which every level's code reads in its own
	// context: written only into a run's own, a native callee's charCodeAt
	// left native code at every call -- Crypto's intAt, until its callers
	// stopped calling it natively, and then Go ran each of its calls.
	// The function's object, or nil, so that the struct compares: a
	// Value's word is a NaN, unequal to itself. strProto, strShape and
	// ccOffset are where it is in String.prototype (abi.Context's
	// StringProto), when it is there.
	charCodeAt *Object
	strProto   *Object
	strShape   *shape
	ccOffset   uint64
}

// jitShareContexts writes what the contexts share into context idx, which
// code is entered in, and every one past it, which its native calls run
// in, when it is not what they hold: a native call then writes none of it.
// They hold it from the last time it was written (jitSharedContexts), if
// that was from idx or before, as nothing else writes it. How deep the
// calls may go from idx, which the VM's depth at the entry decides, is
// written once for them all (abi.Bounds).
func (r *Runtime) jitShareContexts(idx int, f *frame) {
	s := r.jit
	// The level past the deepest the VM's frames allow, which the contexts
	// made so far may limit further: they grow (jitGrowContexts) to it.
	s.levelBound = uint64(idx + max(0, r.maxFrames-r.frameDepth))
	bounds := abi.Bounds{
		LevelLimit: min(uint64(len(s.ssaCtxs)), s.levelBound),
		// The level whose frame is tailCallDepth deep, the run's frame at
		// idx.
		TailLevel: uint64(idx + max(0, tailCallDepth-r.frameDepth)),
	}
	if bounds != s.bounds {
		s.bounds = bounds
		s.ssaSharedGen++
	}
	sh := jitShared{
		global: unsafe.Pointer(f.cl.scope()), lexNames: unsafe.Pointer(&r.lexNames),
		stackBase: unsafe.Pointer(unsafe.SliceData(r.stack)), stackEnd: r.jitStackEnd(),
	}
	if r.isGlobalScope(f.cl.scope()) {
		// The interpreter reads a script's lexical bindings where the
		// scope is the realm's (globalLexProp).
		sh.globalLex = unsafe.Pointer(r.globalLex)
	}
	if p := r.proto.str; p != nil {
		sh.strMethods = unsafe.Pointer(p)
	}
	if m := r.jitStringMethod(); m.IsObject() {
		sh.charCodeAt = m.Object()
		if p := r.proto.str; p.shape != nil {
			if i := p.findOwn(s.charCodeAtKey); i >= 0 {
				// Its shape remembered: a change of its layout gives it
				// another, which native code's check then misses.
				sh.strProto, sh.strShape = p, remember(p.shape)
				sh.ccOffset = uint64(i) * uint64(jitEncoding.ValueSize)
			}
		}
	}
	if sh == s.ssaShared && s.ssaSharedFrom <= idx {
		return
	}
	for i := idx; i < len(s.ssaCtxs); i++ {
		c := s.ssaCtxs[i]
		c.StackBase, c.StackEnd = sh.stackBase, sh.stackEnd
		c.StackTop, c.StackHigh = &r.stackTop, &r.stackHigh
		c.BackEdges, c.GlobalEpoch, c.TypedEpoch = &r.backEdges, &r.shapes.globalEpoch, &r.typedEpoch
		c.Global, c.LexNames, c.GlobalLex, c.StringMethods = sh.global, sh.lexNames, sh.globalLex, sh.strMethods
		*(*Value)(unsafe.Pointer(&c.CharCodeAt)) = Value{}
		if sh.charCodeAt != nil {
			*(*Value)(unsafe.Pointer(&c.CharCodeAt)) = Obj(sh.charCodeAt)
		}
		c.StringProto, c.StringShape, c.CharCodeAtOffset = unsafe.Pointer(sh.strProto), uint64(uintptr(unsafe.Pointer(sh.strShape))), sh.ccOffset
	}
	s.ssaShared, s.ssaSharedFrom = sh, idx
	s.ssaSharedGen++
}

// jitContexts is how many contexts are made first: one for each runSSA
// running, each native call another (abi.Context.Level); and as many more
// as there are each time they run out, as a stack grows, as deep as the
// VM's frames may go (jitContextsMost). A context is some 17 KB: the first
// jitContexts are of the JIT's budget (jitBudget), which every runtime
// that runs native code holds; the rest are the native levels' stack, as
// V8's optimized frames are on the machine stack and not in its code
// space, which a recursion takes as it deepens and gives back once it has
// returned (jitShrinkContexts). With at most 96 of them, AWFY's Havlak,
// whose depth-first search recurses deeper, ran its deeper levels in the
// interpreter, and EarleyBoyer's unifier had finished most of its calls in
// Go with no more than 24.
const jitContexts = 4

// jitContextsMost is how many contexts the runtime's native levels may
// take: one for each frame the VM allows (Config.MaxCallDepth), as each
// level is a frame of the VM's to the depth limit (abi.Bounds'
// LevelLimit), and one past the deepest, which a run Go begins there
// needs (runSSA). A call past the limit is Go's, which throws the
// interpreter's RangeError.
func (r *Runtime) jitContextsMost() int { return r.maxFrames + 2 }

// jitGrowTo is s with room for n elements, its length as it is.
func jitGrowTo[T any](s []T, n int) []T {
	if cap(s) >= n {
		return s
	}
	return append(make([]T, 0, n), s...)
}

// jitGrowContexts adds a chunk of contexts, linked to the last --
// jitContexts, or as many as there are -- if there are fewer than the
// VM's frames need (jitContextsMost), and reports whether it did: the
// first chunk if the JIT's budget has the room for it, the others
// whatever the budget holds, as a deeper recursion needs them. The others
// stay where they are, and native code may be running in them, its
// callers' ReturnTo and Prev pointing at them: each one whose calls the
// contexts limited may go as deep as the new ones allow, and each new one
// holds what they share (jitShareContexts).
func (r *Runtime) jitGrowContexts() bool {
	s := r.jit
	old := len(s.ssaCtxs)
	add := min(max(old, jitContexts), r.jitContextsMost()-old)
	if add <= 0 {
		return false
	}
	if old < jitContexts {
		s.budget = r.jitBudget()
		if !s.makeRoom(add * (jitContextShare + s.spillShare())) {
			return false
		}
	}
	// Each context's places in what is kept by context, made here in the
	// room made for them: grown as runs reach them, they took room no one
	// had made.
	s.ssaCtxs = jitGrowTo(s.ssaCtxs, old+add)
	s.goRuns = jitGrowTo(s.goRuns, old+add)
	s.calleeOpen = jitGrowTo(s.calleeOpen, old+add)
	chunk := make([]abi.Context, add)
	if s.charStrings == nil {
		s.charStrings = new([256]Value)
		for b := range s.charStrings {
			if b < 128 {
				s.charStrings[b] = Str(r.asciiCharString(byte(b)))
			} else {
				s.charStrings[b] = Str(r.latin1CharString(byte(b)))
			}
		}
	}
	for i := range chunk {
		chunk[i].Host = unsafe.Pointer(r)
		chunk[i].CharStrings = unsafe.Pointer(s.charStrings)
		chunk[i].SetConsts(&jitEncoding)
		chunk[i].Level, chunk[i].Bounds, chunk[i].Hints = uint64(old+i), &s.bounds, &s.hintTable
		if s.mega == nil {
			s.mega = new(abi.MegaCache)
		}
		chunk[i].Mega = s.mega
		chunk[i].GrowSpill(s.ssaSpills)
		s.ssaCtxs = append(s.ssaCtxs, &chunk[i])
	}
	for i := max(old, 1); i < len(s.ssaCtxs); i++ {
		c, prev := s.ssaCtxs[i], s.ssaCtxs[i-1]
		prev.Next, c.Prev = unsafe.Pointer(c), unsafe.Pointer(prev)
	}
	s.ctxMost = max(s.ctxMost, len(s.ssaCtxs))
	if old == 0 {
		return true
	}
	n := uint64(len(s.ssaCtxs))
	last := s.ssaCtxs[old-1]
	for _, c := range s.ssaCtxs[old:] {
		c.StackBase, c.StackEnd, c.StackTop, c.StackHigh = last.StackBase, last.StackEnd, last.StackTop, last.StackHigh
		c.BackEdges, c.Global, c.LexNames, c.GlobalLex = last.BackEdges, last.Global, last.LexNames, last.GlobalLex
		c.GlobalEpoch, c.TypedEpoch = last.GlobalEpoch, last.TypedEpoch
		c.StringMethods = last.StringMethods
		c.CharCodeAt = last.CharCodeAt
		c.StringProto, c.StringShape, c.CharCodeAtOffset = last.StringProto, last.StringShape, last.CharCodeAtOffset
	}
	if s.bounds.LevelLimit == uint64(old) && s.levelBound > uint64(old) {
		// Calls the contexts limited may go as deep as the new ones allow,
		// to the VM's frames' limit.
		s.bounds.LevelLimit = min(n, s.levelBound)
		s.ssaSharedGen++
	}
	return true
}

// jitShrinkContexts gives back the chunks of contexts (jitGrowContexts)
// past every one native code runs in now and every one it has run in
// since the last look, a flush's tick before (flush), as Go shrinks a
// goroutine's stack that has used less than it holds: a recursion that
// went deep once keeps none of its contexts, one that recurses as deep
// again within a tick keeps them all. One runs now below ctxTop, which
// a run Go begins, and a call of Go from native code, moves past every
// level it is under; or is a native level past it a call has made
// (abi.Context's Live), which Go has yet to finish or read. One has run
// since if its frame's locals were set (Locals), which a run's entry and
// every native call's callee (mir's callEntry) set, and which the look
// clears in every context past those running, for the next to see. What
// stays is a chunk's end, one past the deepest kept, for a run begun
// there; none of what native code runs in moves.
func (s *jitState) jitShrinkContexts() {
	n := len(s.ssaCtxs)
	if n <= jitContexts {
		return
	}
	keep := s.ctxTop + 1
	for i := s.ctxTop; i < n; i++ {
		c := s.ssaCtxs[i]
		if c.Live != 0 {
			keep = max(keep, i+2)
			continue
		}
		if c.Locals != nil {
			keep = max(keep, i+2)
			c.Locals = nil
		}
	}
	// The chunks end at jitContexts doubled, and at the last's end.
	end := jitContexts
	for end < keep {
		end *= 2
	}
	if end >= n {
		return
	}
	for i := end; i < n; i++ {
		s.ssaCtxs[i] = nil
	}
	s.ssaCtxs[end-1].Next = nil
	s.ssaCtxs = append([]*abi.Context(nil), s.ssaCtxs[:end]...)
	s.goRuns = jitShrinkTo(s.goRuns, end, s.ctxTop == 0)
	s.regRuns = jitShrinkTo(s.regRuns, end, s.ctxTop == 0)
	s.calleeOpen = jitShrinkTo(s.calleeOpen, end, s.ctxTop == 0)
	if s.bounds.LevelLimit > uint64(end) {
		s.bounds.LevelLimit = uint64(end)
		s.ssaSharedGen++
	}
	if s.ssaSharedFrom > end {
		s.ssaSharedFrom = end
	}
	if c := s.ssaCtx; c != nil && c.Level >= uint64(end) {
		// The context last entered, a run's that has ended: let go with
		// its chunk.
		s.ssaCtx = s.ssaCtxs[0]
	}
	s.ctxShrunk++
}

// jitShrinkTo is s with no more than n elements, what it held past them
// -- a run's by context, gone with the contexts (jitShrinkContexts) --
// cleared, in an array of its own if move says no run is under way: a run
// keeps a pointer to its own element across the calls of Go it makes.
func jitShrinkTo[T any](s []T, n int, move bool) []T {
	if len(s) > n {
		clear(s[n:])
		s = s[:n]
	}
	if !move || cap(s) <= n {
		return s
	}
	return append(make([]T, 0, n), s...)
}

// jitNativeLevel is a frame a native call made whose callee left native
// code, or an inlined callee's an exit inside it wrote (abi.Context.Live):
// what Go makes the VM's frame from, and finishes it with. An inlined
// callee's has its program's locals and its receiver's slot plus one
// (abi.Context's Inline fields), and top, the stack's top before it.
type jitNativeLevel struct {
	cl                    *closure
	base                  int
	this                  Value
	kind, pc, depth, site uint64
	inline                bool
	locals, thisSlot, top int
	// args, for an inlined callee's that keeps its call's arguments
	// (abi.Context's InlineArgs), says where they are; 0 for none.
	args uint64
	// prologue, for an inlined callee's that left inside its parameter
	// list's prologue, is its call's arguments, which its frame is given
	// (jitPrologueArgs).
	prologue []Value
	// construct marks a callee a construction called, whose result, if not
	// an object, is its receiver, and newTarget is then the function the
	// construction called, the frame's new.target.
	construct bool
	newTarget Value
	// callee, for an inlined callee's, is its function object
	// (abi.Context's InlineCallee), the frame's, as a stack trace names it
	// by; nil for a native call's.
	callee *Object
	// ctx is the level's context's index.
	ctx int
	// returnTo is where its native caller goes on after the call
	// (abi.Context.ReturnTo), and code the caller's code then: the caller
	// is resumed there once Go has finished this level (jitUnwindNative).
	returnTo uintptr
	code     *jit.SSACode
	// owner, for an inlined callee's level, is the entry whose code
	// inlined it, the code its exit left (jitFinishExit), and ownerCl a
	// closure of its function's.
	owner   *jitEntry
	ownerCl *closure
	// loop is the loop of a built-in's call the level's code was making at
	// its call or exit (abi.Context's Loop), which Go goes on with there;
	// loopFrame says Go made the built-in's frame past the level's
	// (jitPushLoopFrame), its callee's caller.
	loop      uint64
	loopFrame bool
	// slotsFrom and slotsTo are the words its records wrote over, in
	// jitState's levelSlots (jitKeepSlots).
	slotsFrom, slotsTo int
	// gen, for a generator a loop's step resumed natively (jitGenLevel),
	// is the generator, and cursor the step's cursor; fill marks a
	// destructuring's step, whose done is undefined (jitGenDone).
	gen    *generator
	cursor *iterState
	fill   bool
}

// jitDerivedResult is a derived construction's result, from what its
// frame's run gave, as constructWithTarget makes it: an object; undefined
// returned before super(), a ReferenceError; anything else, a TypeError,
// thrown at the construction, which the constructor's own try does not
// catch.
func (r *Runtime) jitDerivedResult(v Value, err error) (Value, error) {
	switch {
	case err == errNoSuper:
		return Undefined, r.throwError(errReference, "%s", errNoSuper.Error())
	case err != nil:
		return Undefined, err
	case !v.IsObject():
		return Undefined, r.throwTypeError(errDerivedReturn)
	}
	return v, nil
}

// jitDerivedFrame gives f, a frame Go makes for a derived constructor's
// level, which a native construction called (ssa.CallSite's Derived), what
// runFD gives a derived construction's frame, from its context: its
// function (abi.Context's Function), new.target, instance, and a binding
// of its this, bound if its super() bound it (This).
func jitDerivedFrame(f *frame, c *abi.Context) {
	if c.Function == 0 || c.NewTarget == 0 {
		return
	}
	f.callee = *(**Object)(unsafe.Pointer(&c.Function))
	f.newTarget = Obj(*(**Object)(unsafe.Pointer(&c.NewTarget)))
	if this := *(*Value)(unsafe.Pointer(&c.This)); !this.IsUninitialized() {
		f.this, f.thisRef = this, &thisBinding{value: this, init: true}
		return
	}
	f.this = *(*Value)(unsafe.Pointer(&c.Instance))
	f.thisRef = &thisBinding{value: f.this}
}

// jitSuperBound binds f's this to what a super() native code made gave
// (ssa.CallSite's Super), its parent finished by Go: its result if an
// object, else the instance, as superCall binds it -- a second super()
// native code does not make (mir's superGuards), and the class has no
// fields for it to give (jitSuperTarget). The binding is the result.
func jitSuperBound(f *frame, v Value) Value {
	if f.thisRef == nil {
		return v
	}
	if !v.IsObject() {
		v = f.thisRef.value
	}
	f.this, f.thisRef.value, f.thisRef.init = v, v, true
	return v
}

// jitSuperTarget is the parent d's super() constructs with, natively
// (ssa.CallSite's Super), and the code it runs: d a derived class of no
// fields -- which super() would give the instance -- making no arrow
// before a super() -- whose this its binding is, which native code binds
// in the context alone (jitArrowBeforeSuper) -- its
// prototype p a constructor of the realm's script, a base one of no
// fields, or derived; p a class with no constructor of its own, the code
// of the one it extends, skip, as V8 skips a default derived constructor
// (FindNonDefaultConstructorOrConstruct). nil where there is none.
func (r *Runtime) jitSuperTarget(d *Object) (p, skip *Object, cl *closure) {
	dfd := d.fn()
	if dfd == nil || dfd.closure == nil || dfd.ctorKind != ctorDerived || dfd.superCtor() != d || dfd.fieldInit() != nil || jitArrowBeforeSuper(dfd.closure.fn) {
		return nil, nil, nil
	}
	code := func(o *Object) *closure {
		fd := o.fn()
		if fd == nil || fd.closure == nil || fd.native != nil || fd.bound || fd.closure.realm != r.Realm || o.class != ClassFunction ||
			isGeneratorTemplate(fd.closure.fn) || fd.fieldInit() != nil {
			return nil
		}
		switch fd.ctorKind {
		case ctorBase:
			return fd.closure
		case ctorDerived:
			if fd.superCtor() == o {
				return fd.closure
			}
		}
		return nil
	}
	if p = d.proto(); p == nil {
		return nil, nil, nil
	}
	if cl = code(p); cl == nil {
		return nil, nil, nil
	}
	if cl.fn.Kind == bytecode.KindDerivedConstructor && jitForwardsSuper(cl.fn) {
		if q := p.proto(); q != nil {
			if qc := code(q); qc != nil && !(qc.fn.Kind == bytecode.KindDerivedConstructor && jitForwardsSuper(qc.fn)) {
				return p, q, qc
			}
		}
		return nil, nil, nil
	}
	return p, nil, cl
}

// jitArrowBeforeSuper reports whether fn, a derived constructor, may make
// an arrow before a super() of its runs: an arrow shares the constructor's
// this binding, which a super() native code makes binds in the context
// (its This), not in a binding of the VM's an arrow made before shares. No
// arrow at a PC before a super()'s, and no loop back from one to a
// super(), and every arrow fn makes is made once any super() it runs has
// run: Go makes it then on a binding bound (jitBindThis), or, on a native
// callee's frame, of the this its context holds (jitCalleeMakes).
func jitArrowBeforeSuper(fn *bytecode.Function) bool {
	lastSuper, firstArrow := -1, len(fn.Code)
	for pc, in := range fn.Code {
		switch in.Op {
		case bytecode.OpSuperCall:
			lastSuper = pc
		case bytecode.OpClosure:
			if int(in.A) < len(fn.Constants) {
				if k := fn.Constants[in.A]; k.Kind == bytecode.ConstFunction && k.Fn != nil && k.Fn.Kind == bytecode.KindArrow {
					firstArrow = min(firstArrow, pc)
				}
			}
		}
	}
	if firstArrow == len(fn.Code) || lastSuper < 0 {
		return false
	}
	if firstArrow < lastSuper {
		return true
	}
	for pc, in := range fn.Code {
		switch in.Op {
		case bytecode.OpJump, bytecode.OpJumpIfFalse, bytecode.OpJumpIfTrue, bytecode.OpJumpIfCmpFalse:
			if pc >= firstArrow && int(in.A) <= lastSuper {
				return true
			}
		}
	}
	return false
}

// jitBindThis binds the this of f, a derived constructor's frame Go
// entered its code with, as a super() native code made bound it (its
// context's This), where Go has not: as the interpreter's super() binds
// the frame's binding, which a closure Go makes on f, an arrow, shares.
// Native code made such a super() only where no arrow is made before it
// (jitArrowBeforeSuper): none shares the binding unbound.
func jitBindThis(f *frame, ctx *abi.Context) {
	if f.thisRef == nil || f.thisRef.init || f.cl.fn.Kind != bytecode.KindDerivedConstructor {
		return
	}
	if this := *(*Value)(unsafe.Pointer(&ctx.This)); !this.IsUninitialized() {
		f.this, f.thisRef.value, f.thisRef.init = this, this, true
	}
}

// jitSuperCall is superCall for a super() Go makes for native code
// (abi.GoCallSuper).
func (r *Runtime) jitSuperCall(f *frame, args []Value) error {
	return r.superCall(f, args)
}


// jitStartLocals makes the locals of a frame a native call made for fn, its
// code not run (abi.ExitEnter), undefined past its parameters, as fn's
// code's call entry would have, which the call left to it (mir's
// callEntry).
func jitStartLocals(fn *bytecode.Function, locals []Value) {
	for i := min(fn.ParamCount, len(locals)); i < len(locals); i++ {
		locals[i] = Undefined
	}
}

// jitShapeMiss reports whether level l, an inlined callee's, left native
// code at a property site for a shape the code was not compiled for, or at
// a site that knew none, its cache not megamorphic: the code is compiled
// again for what the cache learns there (jitFinishExit), and the exit is
// not the callee's leaving native code, which would stop it being inlined
// (jitInlineLeft) -- a callee inlined before its sites met their shapes
// left once for each.
func (r *Runtime) jitShapeMiss(l *jitNativeLevel) bool {
	if !l.inline || int(l.pc) >= len(l.cl.fn.Code) || !jitPropertyOp(l.cl.fn.Code[l.pc].Op) {
		return false
	}
	return l.kind == abi.ExitDeopt || l.kind == abi.ExitHost && !r.jitSiteMegamorphic(l.cl, r.jit.entryFor(l.cl.fn), int(l.pc))
}

// jitPoolEmptied reports whether level l left native code at a construction
// its code makes from a pool, now empty: it left for the pool to be filled
// again (jitRefillPools).
func (r *Runtime) jitPoolEmptied(l *jitNativeLevel) bool {
	fn := l.cl.fn
	if l.kind != abi.ExitHost || int(l.pc) >= len(fn.Code) || fn.Code[l.pc].Op != bytecode.OpNew {
		return false
	}
	e := r.jitEntryOf(l.cl)
	if e == nil {
		return false
	}
	for _, x := range jitTargets(e) {
		if int(x.pc) == int(l.pc) && x.pool != nil && x.pool.Count == 0 {
			return true
		}
	}
	return false
}

// jitTargets iterates over the calls e's code makes natively and those it
// inlines, each list in turn.
func jitTargets(e *jitEntry) iter.Seq2[int, *jitInline] {
	return func(yield func(int, *jitInline) bool) {
		i := 0
		for _, list := range [2][]jitInline{e.nativeCalls, e.inlines} {
			for k := range list {
				if !yield(i, &list[k]) {
					return
				}
				i++
			}
		}
	}
}

// jitUnwindProbe is how many times native code called natively leaves
// between looks at whether it leaves too often, and jitUnwindShare the
// share of its calls it may leave from, as one in so many: Go then finishes
// every native call outside it (jitUnwindNative), which costs more than Go
// making the call.
const (
	jitUnwindProbe = 64
	jitUnwindShare = 4
)

// jitUnwound counts a native call that left native code inside e's code
// (nativeOut). Code that leaves on more than one in jitUnwindShare of its
// calls is compiled again with what its exits taught since (jitCallSeen,
// jitPolySeen), its native callers calling it still, as V8 deoptimizes
// code and optimizes it again with the feedback gathered since: leaving
// while it learned, its calls and constructions not yet decided, is no
// reason to stop calling it natively. Only once those compiles are spent
// (jitUnwindReopts), as V8 gives up optimizing a function that deoptimizes
// too often, do native callers stop calling it and leave for Go at the
// call instead, which makes it (notNative), until jitRetryNative -- or at
// once, if its exits taught nothing since it was compiled (jitLearning),
// for code compiled again would be the same: RayTrace's getReflectionRay,
// constructing with a forwarding constructor its code leaves to Go, and
// wrapUp, whose remainder of fractions Go computes, were compiled three
// times more each for nothing.
func (r *Runtime) jitUnwound(e *jitEntry) {
	if e.nativeOut++; e.nativeOut%jitUnwindProbe != 0 || e.nativeOut*jitUnwindShare <= e.link.in {
		return
	}
	if e.ssa != nil && e.unwindReopts < jitUnwindReopts && jitLearning(e) {
		e.unwindReopts++
		e.unwindReopt = true
		e.link.in, e.nativeOut = 0, 0
		e.ssaStats.calleeGoCalls, e.ssaStats.calleeWork = 0, 0
		return
	}
	e.notNative, e.nativeRetry = true, 0
	e.link.clearCode()
	e.nativeBackoff = min(e.nativeBackoff+1, jitNativeBackoffs)
}

// jitLearning reports whether e's code has met, since it was compiled,
// what a compile would take that it was not compiled for: what is to be
// compiled again for (reoptPending), calls its exits learned
// (inlinePending), or a call it has left native
// code at but not yet decided, which the compile would decide if the
// existing tiers have met its target (jitSeedCalls).
func jitLearning(e *jitEntry) bool {
	if e.reoptPending() || e.inlinePending {
		return true
	}
	for _, n := range e.callSites {
		if n != 0 && n < jitCallWaiting {
			return true
		}
	}
	return false
}

// jitUnwindReopts is how many times code that leaves too often when called
// natively is compiled again for it before native callers stop calling it
// (jitUnwound).
const jitUnwindReopts = 3

// jitNativeRetry is how many calls Go makes to code native callers no
// longer call before they call it again, doubled for each time it was
// found to leave too often, up to jitNativeBackoffs times: code that left
// while it learned what to compile for, or whose callers' objects changed,
// may not now.
const (
	jitNativeRetry    = 1024
	jitNativeBackoffs = 8
)

// jitRetryNative counts a call Go makes to e's code, which native callers
// no longer call (notNative) -- one Go enters it for, or one a native
// caller left for Go to make (jitCallTarget), which a function Go does not
// enter, or calls without a frame, has only -- and has them call it again
// after jitNativeRetry, counting its leaving afresh.
func jitRetryNative(e *jitEntry) {
	if !e.owned() || e.invalid {
		// Dropped by the cache: its code is not its function's link's; or
		// invalidated, which native callers call again once compiled
		// again.
		return
	}
	if e.nativeRetry++; e.nativeRetry >= jitNativeRetry<<(e.nativeBackoff-1) {
		e.notNative, e.link.in, e.nativeOut = false, 0, 0
		e.ssaStats.calleeGoCalls, e.ssaStats.calleeWork = 0, 0
		e.link.setCode(e.ssa)
	}
}

// jitTailGivesUp reports whether the instruction at pc of the frame Go
// entered native code with, f, is a call in tail position that gives up the
// frame, as the interpreter's tail_call decides: deep in the stack, in a
// frame no construction's, nor an arrow's sharing a derived constructor's
// this. (The compiler emits none under a handler.)
func (r *Runtime) jitTailGivesUp(f *frame, pc int) bool {
	op := f.cl.fn.Code[pc].Op
	return (op == bytecode.OpTailCall || op == bytecode.OpTailCallMethod) && r.frameDepth >= tailCallDepth &&
		f.newTarget.IsUndefined() && f.thisRef == nil
}

// jitCallOp is a call's opcode as native code makes the call: a call in
// tail position is an ordinary call where the stack is shallow, which is
// the only place native code makes it (ssa.OpTailRoom); deeper, Go makes
// it as the interpreter does (jitHost).
func jitCallOp(op bytecode.Op) bytecode.Op {
	switch op {
	case bytecode.OpTailCall:
		return bytecode.OpCall
	case bytecode.OpTailCallMethod:
		return bytecode.OpCallMethod
	}
	return op
}

// jitCallInstr is in with its opcode as native code makes the call
// (jitCallOp).
func jitCallInstr(in bytecode.Instr) bytecode.Instr {
	in.Op = jitCallOp(in.Op)
	return in
}

// jitCallResult has a frame whose state at the call at pc, depth operands
// deep, is in its frame go on after the call, which returned v or threw
// err: the call's operands go, its result comes, as when Go makes it
// (jitHost). It returns the operands' top, and false for a throw, which
// the interpreter takes from there.
func (r *Runtime) jitCallResult(f *frame, pc, depth int, v Value, err error) (int, bool) {
	if sp, ok, host := r.jitHostResult(f, pc, depth, v, err); host {
		// One Go made from native code on the frame (jitGoHost).
		return sp, ok
	}
	in := jitCallInstr(f.cl.fn.Code[pc])
	if in.Op == bytecode.OpIterStep || in.Op == bytecode.OpIterRest {
		// A destructuring's step Go made (abi.GoCallFill), or its rest
		// element's array (abi.GoCallRest): its cursor stays, and the value
		// goes on top. The rest's array had taken the cursor's place, as a
		// call's result takes its callee's, and every operand after it was
		// one below its slot: r.join('') after const [a, ...q] = xs[i] met
		// a string as its receiver, and undefined as its method.
		sp := f.base + depth
		f.pc = uint32(pc + 1)
		if err != nil {
			return sp, false
		}
		r.stack[sp] = v
		return sp + 1, true
	}
	if in.Op == bytecode.OpIterNextOrJump {
		// A loop's step Go made (abi.GoCallStep): its cursor stays, and the
		// next value goes on it; or, done, the loop's end is next.
		sp := f.base + depth
		f.pc = uint32(pc + 1)
		switch {
		case err != nil:
			return sp, false
		case v.IsUninitialized():
			f.pc = in.A
			return sp, true
		}
		r.stack[sp] = v
		return sp + 1, true
	}
	switch in.Op {
	case bytecode.OpGetProp, bytecode.OpGetPropThis, bytecode.OpGetPropUnder, bytecode.OpSetProp:
		// An accessor's read or write, its getter or setter called
		// (ssa.AccessorSite), or one Go made (abi.GoCallGet): the object
		// goes but for get_prop_this, which keeps it below the value, and
		// an object pattern's get_prop_under, whose source stays where it
		// is; a write's value goes too, and leaves nothing.
		sp := f.base + depth
		f.pc = uint32(pc + 1)
		switch in.Op {
		case bytecode.OpGetProp:
			sp--
		case bytecode.OpSetProp:
			sp -= 2
		}
		if err != nil {
			return sp, false
		}
		if in.Op == bytecode.OpSetProp {
			return sp, true
		}
		r.stack[sp] = v
		return sp + 1, true
	}
	sp := f.base + depth - int(in.A) - 1
	if jitcompile.GoOperation(in.Op) {
		// An operation Go made (abi.GoCallOp): its operands go.
		sp = f.base + depth - jitcompile.GoOperands(in)
	}
	switch in.Op {
	case bytecode.OpTypeOf:
		// Its operand (abi.GoCallTypeOf).
		sp = f.base + depth - 1
	case bytecode.OpInstanceOf:
		// The object and the constructor (abi.GoCallInstanceOf).
		sp = f.base + depth - 2
	case bytecode.OpSuperCall:
		// Its arguments (abi.GoCallSuper); the this it bound comes.
		sp = f.base + depth - int(in.A)
	case bytecode.OpCallMethod:
		sp--
	case bytecode.OpApplyArguments:
		// f, apply and the receiver (a forwarding constructor's inlined
		// frame, ssa.InlineSite's Forward).
		sp = f.base + depth - 3
	}
	f.pc = uint32(pc + 1)
	if err != nil {
		return sp, false
	}
	r.stack[sp] = v
	return sp + 1, true
}

// jitLevelHost reports whether op, an instruction a native callee's code
// has Go make from native code (ssa.GoHost), may be made on the frame of
// its level that Go makes, which has its closure, its receiver, its
// operands and, where its code reads it, its function object (the call
// wrote it, jitFrameCallee), but not its locals, which native code holds:
// a definition's (compile's frameDefinition), a super reference's, a
// mapped arguments object's of a function of no parameters, which aliases
// none.
func jitLevelHost(op bytecode.Op) bool {
	switch op {
	case bytecode.OpGetSuperProp, bytecode.OpGetSuperIndex, bytecode.OpSetSuperProp, bytecode.OpSetSuperIndex, bytecode.OpSuperBase,
		bytecode.OpGetArguments:
		return true
	case bytecode.OpGetLength, bytecode.OpNewClass, bytecode.OpSetHomeObject, bytecode.OpSetFieldInit, bytecode.OpDefineMethod,
		bytecode.OpDefineGetter, bytecode.OpDefineSetter, bytecode.OpSetFuncName,
		bytecode.OpPrivateName, bytecode.OpNewPrivateMethods, bytecode.OpInstallPrivateMethods,
		bytecode.OpDefinePrivate, bytecode.OpDefinePrivateMethod, bytecode.OpDefinePrivateGetter,
		bytecode.OpDefinePrivateSetter, bytecode.OpAddPrivateMethod, bytecode.OpAddPrivateGetter, bytecode.OpAddPrivateSetter:
		return true
	}
	return false
}

// jitHostResult is jitCallResult for an instruction Go made from native
// code on the frame (ssa.GoHost), at pc, with depth operands: its operands
// popped, and its result v, if it pushes one, pushed; false for a throw.
func (r *Runtime) jitHostResult(f *frame, pc, depth int, v Value, err error) (int, bool, bool) {
	in := f.cl.fn.Code[pc]
	pops, pushes := 0, true
	switch in.Op {
	case bytecode.OpRestParam, bytecode.OpGetArguments, bytecode.OpGetSuperProp, bytecode.OpSuperBase,
		bytecode.OpPrivateName, bytecode.OpNewPrivateMethods:
	case bytecode.OpGetSuperIndex:
		pops = 2
	case bytecode.OpGetLength:
		pops = 1
	case bytecode.OpSetSuperProp:
		pops, pushes = 1, false
	case bytecode.OpSetSuperIndex:
		pops, pushes = 3, false
	case bytecode.OpSetHomeObject, bytecode.OpSetFuncName:
		// What they change stays where it is.
		pushes = false
	case bytecode.OpNewClass, bytecode.OpSetFieldInit, bytecode.OpDefineMethod, bytecode.OpDefineGetter, bytecode.OpDefineSetter,
		bytecode.OpDefinePrivate, bytecode.OpDefinePrivateMethod, bytecode.OpDefinePrivateGetter, bytecode.OpDefinePrivateSetter,
		bytecode.OpInstallPrivateMethods:
		// The member, or the parent, popped; the object kept.
		pops, pushes = 1, false
	case bytecode.OpAddPrivateMethod, bytecode.OpAddPrivateGetter, bytecode.OpAddPrivateSetter:
		pops, pushes = 2, false
	default:
		return 0, false, false
	}
	sp := f.base + depth - pops
	f.pc = uint32(pc + 1)
	if err != nil {
		return sp, false, true
	}
	if pushes {
		r.stack[sp] = v
		sp++
	}
	return sp, true, true
}

// jitArgument is ssa.GoArgument's answer, for the code ctx runs, a run of
// a function's own code Go entered with its frame, which has the call's
// arguments: arguments[k] -- the first operand k, the second the local
// the arguments object is kept in once made -- or, of one operand, that
// local, arguments.length; and whether it answers. While the object is
// not made, an index of the arguments is the argument, as the
// interpreter's argumentsIndex reads it -- a mapped object's too, which
// aliases parameters the function never assigns (compile's
// argumentsOwn), or, where it or a closure assigns them, the parameter
// the index aliases, which the code passes past the two operands
// (ir.Program's ArgumentParams) -- and their count is how many there are.
// The rest it leaves to the code's exit, where Go makes it.
func (r *Runtime) jitArgument(ctx *abi.Context) (Value, bool) {
	s, idx := r.jit, int(ctx.Level)
	var args []Value
	var fn *bytecode.Function
	switch {
	case ctx.Live == abi.LiveCall:
		// A native callee's, whose call wrote them to its context
		// (jitFrameArgs).
		cl := (*closure)(ctx.Closure)
		if cl == nil || !jitFrameArgs(cl.fn) {
			return Undefined, false
		}
		args, fn = jitContextArgs(ctx), cl.fn
	case ctx.Live != 0 || idx >= len(s.goRuns) || idx >= len(s.ssaCtxs) || s.ssaCtxs[idx] != ctx || s.goRuns[idx].f == nil:
		return Undefined, false
	default:
		args, fn = s.goRuns[idx].f.args, s.goRuns[idx].f.cl.fn
	}
	if ctx.GoOp>>abi.GoOpArgcShift&0xff == 1 {
		if jitGoArg(ctx, 0).IsObject() {
			return Undefined, false
		}
		return Int(len(args)), true
	}
	k := jitGoArg(ctx, 0)
	if jitGoArg(ctx, 1).IsObject() || !k.IsNumber() {
		return Undefined, false
	}
	if i := uint32(k.num); float64(i) == k.num && uint(i) < uint(len(args)) {
		if n := int(ctx.GoOp>>abi.GoOpArgcShift&0xff) - 2; uint(i) < uint(n) && mapsArgument(fn, int(i)) {
			// The parameter the mapped object would alias, as the code
			// has it now (ir.Program's ArgumentParams): one it, or a
			// closure, assigned.
			return jitGoArg(ctx, 2+int(i)), true
		}
		return args[i], true
	}
	return Undefined, false
}

// jitPrologueArgs is the arguments of the call that made a frame of p's
// code that native code left at pc, depth operands deep, inside its
// parameter list's prologue (ir.Program's PrologueArgs) -- before the
// prologue's ParamsDead, the parameters' slots, locals; after it, the
// operands past depth, from operands on, where the exit's state kept them
// -- or nil where the frame needs none. A frame Go makes of a native
// callee's, or of an inlined one's, has no call's arguments of its own, as
// a frame of the VM's has, which the interpreter, going on in the
// prologue, binds the parameters still in the dead zone from (OpInitParam,
// OpParamNeedsDefault).
func jitPrologueArgs(p *ir.Program, pc, depth int, locals, operands []Value) []Value {
	n, inSlots, ok := p.PrologueAt(pc)
	switch {
	case !ok:
		return nil
	case inSlots && n <= len(locals):
		return slices.Clone(locals[:n])
	case !inSlots && depth >= 0 && depth+n <= len(operands):
		return slices.Clone(operands[depth : depth+n])
	}
	return nil
}


// jitFrameArgs reports whether fn's code reads its call's arguments past
// its parameters, from the frame of the VM's: a rest parameter's, the
// arguments object's, a super() of them (bytecode.Function's
// SuperForwards). A native
// call of it writes them all to its callee's context (ssa.CallSite's
// FrameArgs, abi.Context's Args), where Go finds them for the frames it
// makes of the callee's level, as V8's call pushes all of a call's
// arguments, its callee reading them past its parameters.
func jitFrameArgs(fn *bytecode.Function) bool {
	return fn.HasRest || fn.UsesArguments || fn.MappedArguments || fn.Kind == bytecode.KindDerivedConstructor && fn.SuperForwards
}

// jitFrameCallee reports whether fn's frame must have its function object
// (bytecode.Function's FrameCallee, which the compiler records): its code
// reads it -- a super reference's home
// object, push_callee's, a mapped arguments object's callee -- or an arrow
// it makes takes the home object or new.target from it. A native call of
// it writes the object to its callee's context (ssa.CallSite's Function,
// abi.Context's Function), where Go finds it for the frames it makes of the
// callee's level, as V8's frame has its function.
func jitFrameCallee(fn *bytecode.Function) bool { return fn.FrameCallee }

// jitCallArgs is a copy of the arguments a native call of fn wrote to its
// callee's context, c (abi.Context's Args), for a frame Go makes of the
// callee's level that outlives what Go does there -- one the interpreter
// finishes -- or a super() that hands them on: fn's code reads them past
// its parameters (jitFrameArgs). nil for any other function.
func jitCallArgs(fn *bytecode.Function, c *abi.Context) []Value {
	if !jitFrameArgs(fn) {
		return nil
	}
	return slices.Clone(jitContextArgs(c))
}

// jitContextArgs is the arguments a native call wrote to its callee's
// context, c (abi.Context's Args), in place: valid while the callee's level
// runs, its next call writing the next context's.
func jitContextArgs(c *abi.Context) []Value {
	n := min(int(c.Argc), len(c.Args))
	return unsafe.Slice((*Value)(unsafe.Pointer(&c.Args[0])), n)
}

// jitEnterArgs is the arguments of a native call of fn whose frame Go
// runs from its start (abi.ExitEnter), where its parameter list has a
// prologue that binds its parameters from them: the parameters' slots, as
// the call wrote them; or nil.
func jitEnterArgs(fn *bytecode.Function, locals []Value) []Value {
	if !fn.ParamsAreLexical || fn.HasRest || fn.ParamCount > len(locals) {
		return nil
	}
	return slices.Clone(locals[:fn.ParamCount])
}

// jitHandlers gives f, a frame native code ran p's code in, the exception
// handlers in force at pc (ir.Program's HandlerAt), as the interpreter's
// push_catch and pop_catch would have left them there: native code
// registers none, and a throw Go unwinds the frame from finds them.
func jitHandlers(f *frame, p *ir.Program, pc int) {
	if p == nil || p.HandlerAt == nil || pc < 0 || pc >= len(p.HandlerAt) {
		return
	}
	n := 0
	for k := p.HandlerAt[pc]; k != 0; k = p.Handlers[k-1].Outer {
		n++
	}
	f.handlers = slices.Grow(f.handlers[:0], n)[:n]
	for k, i := p.HandlerAt[pc], n-1; k != 0; k, i = p.Handlers[k-1].Outer, i-1 {
		h := p.Handlers[k-1]
		f.handlers[i] = handler{pc: h.PC, stackDepth: int(h.Depth), isFinally: h.Finally}
	}
}

// jitSuspend suspends f, a generator's or an async function's frame, at
// in, a yield, a yield* or an await native code left it at, with sp its
// stack's top,
// as the interpreter's yield does: the value handed out popped, the pc
// past the instruction, and the depth beneath it kept for the resumption,
// which enters native code there (jitResume). The suspension is the
// runtime's own, as the tree tier's is (treeSig).
func (r *Runtime) jitSuspend(f *frame, in bytecode.Instr, sp int) (Value, error, bool) {
	sp--
	f.pc++
	f.savedSP = sp
	r.treeSig = suspendSignal{value: r.stack[sp], await: in.Op == bytecode.OpAwait,
		delegate: in.Op == bytecode.OpYieldStar, raw: in.Op == bytecode.OpYieldStar && in.A == 1}
	return Undefined, &r.treeSig, true
}

// jitCatch unwinds f, a frame of e's code, for err, which what Go did at
// an exit threw with the operands' top at sp, as the interpreter does
// (unwindToHandler): where a handler of the function catches it and e's
// code has an entry there -- every handler has one -- native code goes on
// from it (true), as V8's code goes on at a handler. Otherwise the
// interpreter takes the frame from what it returns: unwound, with no
// error, where a handler caught it; as it was where none did.
func (r *Runtime) jitCatch(f *frame, e *jitEntry, sp int, err error) (int, error, bool) {
	if err == nil || r.stopped != nil || len(f.handlers) == 0 {
		return sp, err, false
	}
	sp, caught := r.unwindToHandler(f, sp, err)
	if !caught {
		return sp, err, false
	}
	return sp, nil, !e.entrySlow && e.ssa.HasEntry(int(f.pc))
}

// jitTypeWord is the type word an interface holding x has (abi.Encoding's
// MapIterData): what native code compares an object's data with to know
// what it holds.
// The element types native code reads are the VM's: either difference
// overflows, which does not compile, unless they are the same.
const (
	_ = uint8(elemInt8-abi.ElemInt8) + uint8(abi.ElemInt8-elemInt8)
	_ = uint8(elemUint8Clamped-abi.ElemUint8Clamped) + uint8(abi.ElemUint8Clamped-elemUint8Clamped)
	_ = uint8(elemUint16-abi.ElemUint16) + uint8(abi.ElemUint16-elemUint16)
	_ = uint8(elemUint32-abi.ElemUint32) + uint8(abi.ElemUint32-elemUint32)
	_ = uint8(elemFloat16-abi.ElemFloat16) + uint8(abi.ElemFloat16-elemFloat16)
	_ = uint8(elemFloat64-abi.ElemFloat64) + uint8(abi.ElemFloat64-elemFloat64)
	_ = uint8(elemBigInt64-abi.ElemBigInt64) + uint8(abi.ElemBigInt64-elemBigInt64)
	_ = uint8(elemBigUint64-abi.ElemBigUint64) + uint8(abi.ElemBigUint64-elemBigUint64)
)

func jitTypeWord(x any) uint64 {
	return uint64((*[2]uintptr)(unsafe.Pointer(&x))[0])
}

// jitForgetHighSeen is jitStackHighLowered: Go has lowered the VM stack's
// high water, and each context forgets what it last found it at
// (abi.Context's HighSeen), which a call entry trusts never to be above
// it; their next call entries raise it again where their frames reach
// past it.
func jitForgetHighSeen(r *Runtime) {
	if r.jit == nil {
		return
	}
	for _, c := range r.jit.ssaCtxs {
		c.HighSeen = 0
	}
}
