package jit

import (
	"runtime"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// MaxDispatchPrograms and MaxDispatchFrames bound native dispatch ownership
// and nesting. A denied transfer exits at its original, uncommitted call.
const (
	MaxDispatchPrograms = 8
	MaxDispatchFrames   = 8
)

// DispatchCall grants one monomorphic transfer. Function and Receiver are
// encoded identities owned by the caller's shared reference table. Receiver
// is checked only for method calls; an ordinary call uses the target template.
type DispatchCall struct {
	PC       int
	Target   int
	Function ir.Value
	Receiver ir.Value
}

// DispatchProgram describes an exclusively borrowed executable and its fresh
// scalar frame template. Params counts copied parameters; StackCharge is the
// embedding VM's canonical stack reservation for one invocation.
type DispatchProgram struct {
	Code        *Code
	Template    []ir.Value
	Params      int
	StackCharge uint64
	Calls       []DispatchCall
}

type nativeDispatchProgram struct {
	entry    *byte
	code     *byte
	entries  *int
	template *ir.Value
	slots    uint64
	params   uint64
	charge   uint64
	sites    uint64
}

type nativeDispatchSite struct {
	program      uint64
	functionSlot uint64
	argSlot      uint64
	argc         uint64
	result       uint64
	receiverSlot int64
	function     ir.Value
	receiver     ir.Value
	_            [6]uint64
}

type nativeDispatchFrame struct {
	program    uint64
	pc         uint64
	result     uint64
	argSlot    uint64
	argc       uint64
	charge     uint64
	steps      uint64
	lastBudget uint64
	invocation uint64
	_          [7]uint64
}

// The original 40-byte program ABI remains the prefix. Only dispatch-enabled
// code and its bridge access the extension. Native stores change scalars only;
// immutable typed pointers keep all borrowed buffers visible to Go's GC.
type dispatchState struct {
	programState
	depth        uint64
	programCount uint64
	ticks        uint64
	stackUsed    uint64
	stackLimit   uint64
	calls        uint64
	frameLimit   uint64
	programs     *nativeDispatchProgram
	frames       *nativeDispatchFrame
	slots        *ir.Value
	sites        *nativeDispatchSite
}

// Dispatch owns bounded scalar frames and prepared native transfers. It never
// owns executable mappings: Reset releases its borrows, and Code.Close remains
// the executable owner's responsibility. Concurrent use is unsupported.
type Dispatch struct {
	state     dispatchState
	owners    [MaxDispatchPrograms]*Code
	programs  [MaxDispatchPrograms]nativeDispatchProgram
	sites     [MaxDispatchPrograms][ir.MaxCallSites]nativeDispatchSite
	templates [MaxDispatchPrograms][ir.MaxSlots]ir.Value
	frames    [MaxDispatchFrames]nativeDispatchFrame
	slots     [MaxDispatchFrames][ir.MaxSlots]ir.Value
	begun     bool
}

// DispatchExit identifies the active program and frame at the ordinary exit.
// Exit.Steps includes committed calls and callee instructions. Calls and Ticks
// are cumulative since Begin and let a VM preserve its call interrupt cadence.
type DispatchExit struct {
	Exit      ir.Exit
	Program   int
	Depth     int
	Calls     uint64
	Ticks     uint64
	StackUsed uint64
}

// DispatchFrame reports a live frame. Suspended PC is the instruction after
// its committed call, with a result still pending at the child's ResultSlot.
// ArgSlot/ArgCount refer to unmodified cells in the parent's scalar buffer.
type DispatchFrame struct {
	Program    int
	PC         int
	ResultSlot int
	ArgSlot    int
	ArgCount   int
	Steps      uint64
	Invocation uint64
}

// DispatchSnapshot supplies a materialized scalar frame to Restore. Values may
// borrow the corresponding buffer returned by Frame on this Dispatch.
type DispatchSnapshot struct {
	Frame  DispatchFrame
	Values []ir.Value
}

func validDispatchScalar(v ir.Value) bool {
	return v.Kind <= ir.Opaque && (v.Kind != ir.Boolean || v.Bits <= 1)
}

func (c *Code) dispatchOffset(pc int) (int, error) {
	if c == nil || len(c.code) == 0 {
		return 0, ErrClosed
	}
	if !c.dispatch || pc < 0 || pc >= len(c.dispatchEntries) || c.dispatchEntries[pc] == -1 {
		return 0, ir.ErrState
	}
	return c.dispatchEntries[pc], nil
}

// Prepare validates every owner, template and grant before replacing the
// graph. A failed preparation preserves the old graph and its scalar frames.
// A successful preparation discards previous execution state; Begin starts it.
func (d *Dispatch) Prepare(programs []DispatchProgram) error {
	if err := validateDispatchPrograms(programs); err != nil {
		return err
	}
	d.prepare(programs)
	return nil
}

func validateDispatchPrograms(programs []DispatchProgram) error {
	if len(programs) == 0 || len(programs) > MaxDispatchPrograms {
		return ir.ErrState
	}
	for _, p := range programs {
		if _, err := p.Code.dispatchOffset(0); err != nil {
			return err
		}
		if len(p.Template) != p.Code.slots || p.Params < 0 || p.Params > p.Code.locals || len(p.Calls) > ir.MaxCallSites {
			return ir.ErrState
		}
		for _, value := range p.Template {
			if !validDispatchScalar(value) {
				return ir.ErrState
			}
		}
		var seen [ir.MaxCallSites]bool
		for _, call := range p.Calls {
			if call.Target < 0 || call.Target >= len(programs) || call.Function.Kind != ir.Opaque || !validDispatchScalar(call.Receiver) {
				return ir.ErrState
			}
			index := -1
			for i, entry := range p.Code.calls {
				if entry.pc == call.PC {
					index = i
					break
				}
			}
			if index < 0 || seen[index] {
				return ir.ErrState
			}
			seen[index] = true
		}
	}
	return nil
}

func (d *Dispatch) prepare(programs []DispatchProgram) {
	d.Reset()
	for i, p := range programs {
		c := p.Code
		d.owners[i] = c
		copy(d.templates[i][:], p.Template)
		n := &d.programs[i]
		*n = nativeDispatchProgram{
			code: &c.code[0], entries: &c.dispatchEntries[0],
			template: &d.templates[i][0], slots: uint64(c.slots), params: uint64(p.Params),
			charge: p.StackCharge, sites: uint64(len(c.calls)),
		}
		if entry := c.dispatchEntries[0]; entry >= 0 {
			n.entry = &c.code[entry]
		}
		for j, entry := range c.calls {
			in := entry.in
			d.sites[i][j] = nativeDispatchSite{
				program: ^uint64(0), functionSlot: uint64(in.Left.Slot), argSlot: uint64(in.Third.Slot),
				argc: uint64(in.Extra), result: uint64(in.Dest), receiverSlot: int64(in.Right.Slot),
			}
		}
		for _, call := range p.Calls {
			for j, entry := range c.calls {
				if entry.pc == call.PC {
					site := &d.sites[i][j]
					site.program, site.function, site.receiver = uint64(call.Target), call.Function, call.Receiver
					break
				}
			}
		}
	}
	d.state.programCount = uint64(len(programs))
	d.state.programs, d.state.frames = &d.programs[0], &d.frames[0]
	d.state.slots, d.state.sites = &d.slots[0][0], &d.sites[0][0]
}

// Restore refreshes the executable graph and all live scalar frames after host
// work. It validates suspended call layouts independently of future grants: a
// callback may replace a binding while an older callee is still active. Calls
// and invocation identities remain monotonic across restoration. The embedding
// VM owns closure identities, reference roots and canonical extra arguments.
// Invalid input preserves the previous graph and execution state.
func (d *Dispatch) Restore(programs []DispatchProgram, snapshots []DispatchSnapshot, frameLimit int, stackLimit, ticks, calls uint64) error {
	if err := validateDispatchPrograms(programs); err != nil {
		return err
	}
	if len(snapshots) == 0 || frameLimit < len(snapshots) || frameLimit > MaxDispatchFrames {
		return ir.ErrState
	}
	var used, invocation uint64
	for i, snapshot := range snapshots {
		f := snapshot.Frame
		if f.Program < 0 || f.Program >= len(programs) {
			return ir.ErrState
		}
		p := programs[f.Program]
		if _, err := p.Code.dispatchOffset(f.PC); err != nil {
			return err
		}
		if len(snapshot.Values) != p.Code.slots {
			return ir.ErrState
		}
		for _, v := range snapshot.Values {
			if !validDispatchScalar(v) {
				return ir.ErrState
			}
		}
		if i == 0 {
			if f.Invocation != 0 || f.ResultSlot != 0 || f.ArgSlot != 0 || f.ArgCount != 0 {
				return ir.ErrState
			}
			continue
		}
		if f.Invocation <= invocation || f.Invocation > calls || p.StackCharge > stackLimit-used {
			return ir.ErrState
		}
		invocation = f.Invocation
		used += p.StackCharge
		parent := snapshots[i-1].Frame
		var call *ir.Instruction
		for j := range programs[parent.Program].Code.calls {
			entry := &programs[parent.Program].Code.calls[j]
			if entry.pc == parent.PC-1 {
				call = &entry.in
				break
			}
		}
		if call == nil || f.ResultSlot != call.Dest || f.ArgSlot != call.Third.Slot || f.ArgCount != call.Extra {
			return ir.ErrState
		}
	}
	d.prepare(programs)
	for i, snapshot := range snapshots {
		f := snapshot.Frame
		d.frames[i] = nativeDispatchFrame{
			program: uint64(f.Program), pc: uint64(f.PC), result: uint64(f.ResultSlot),
			argSlot: uint64(f.ArgSlot), argc: uint64(f.ArgCount), steps: f.Steps, invocation: f.Invocation,
		}
		if i != 0 {
			d.frames[i].charge = programs[f.Program].StackCharge
		}
		copy(d.slots[i][:], snapshot.Values)
	}
	d.state.depth, d.state.stackUsed, d.state.calls = uint64(len(snapshots)-1), used, calls
	d.state.frameLimit, d.state.stackLimit, d.state.ticks = uint64(frameLimit), stackLimit, ticks
	d.begun = true
	return nil
}

// Begin copies encoded root state and establishes depth, stack and call-check
// limits. Native transfers cannot exceed any of these independent limits.
func (d *Dispatch) Begin(program, pc int, slots []ir.Value, frames int, stack, ticks uint64) error {
	if program < 0 || uint64(program) >= d.state.programCount || frames < 1 || frames > MaxDispatchFrames {
		return ir.ErrState
	}
	c := d.owners[program]
	if _, err := c.dispatchOffset(pc); err != nil {
		return err
	}
	if len(slots) != c.slots {
		return ir.ErrState
	}
	for _, v := range slots {
		if !validDispatchScalar(v) {
			return ir.ErrState
		}
	}
	clear(d.frames[:])
	copy(d.slots[0][:], slots)
	d.frames[0].program, d.frames[0].pc = uint64(program), uint64(pc)
	d.state.depth, d.state.stackUsed, d.state.calls = 0, 0, 0
	d.state.frameLimit, d.state.stackLimit, d.state.ticks = uint64(frames), stack, ticks
	d.begun = true
	return nil
}

// Frame borrows one active scalar buffer until the next Prepare or Begin.
// Only valid encoded scalars may be written through the returned slice.
func (d *Dispatch) Frame(index int) (DispatchFrame, []ir.Value, error) {
	if !d.begun || index < 0 || uint64(index) > d.state.depth {
		return DispatchFrame{}, nil, ir.ErrState
	}
	f := d.frames[index]
	n := d.programs[f.program].slots
	return DispatchFrame{
		Program: int(f.program), PC: int(f.pc), ResultSlot: int(f.result), ArgSlot: int(f.argSlot),
		ArgCount: int(f.argc), Steps: f.steps, Invocation: f.invocation,
	}, d.slots[index][:n], nil
}

// Resume updates the active frame after committed host work. It preserves all
// suspended callers. The VM must refresh graph permissions before callbacks
// can change their bindings, receivers, templates or executable ownership.
func (d *Dispatch) Resume(pc int) error {
	if !d.begun {
		return ir.ErrState
	}
	f := &d.frames[d.state.depth]
	if _, err := d.owners[f.program].dispatchOffset(pc); err != nil {
		return err
	}
	f.pc = uint64(pc)
	return nil
}

// Run borrows array permissions for one bounded native batch. No Go callback,
// stack change or executable allocation occurs between its call and return.
// All owners are checked before effects, including suspended/possible callees.
func (d *Dispatch) Run(arrays []ir.ArrayView, budget uint64) (DispatchExit, error) {
	if !d.begun || len(arrays) != 0 && len(arrays) != ir.MaxSlots {
		return DispatchExit{}, ir.ErrState
	}
	if budget > MaxIterations {
		return DispatchExit{}, ErrIterations
	}
	for i := uint64(0); i < d.state.programCount; i++ {
		if len(d.owners[i].code) == 0 {
			return DispatchExit{}, ErrClosed
		}
	}
	s := &d.state
	f := &d.frames[s.depth]
	c := d.owners[f.program]
	entry, err := c.dispatchOffset(int(f.pc))
	if err != nil {
		return DispatchExit{}, err
	}
	s.programState = programState{remaining: budget, pc: f.pc}
	f.lastBudget = budget
	if entry == hostProgramEntry {
		s.reason = uint64(ir.HostExit)
		if budget == 0 {
			s.reason = uint64(ir.BudgetExit)
		}
	} else {
		runDispatchCode(&c.code[entry], s, &d.slots[s.depth][0], arrays)
	}
	f = &d.frames[s.depth]
	f.pc = s.pc
	if ir.ExitKind(s.reason) != ir.Returned {
		f.steps += f.lastBudget - s.remaining
	}
	f.lastBudget = s.remaining
	c = d.owners[f.program]
	result := DispatchExit{
		Exit:    ir.Exit{Kind: ir.ExitKind(s.reason), State: c.maps[s.pc], Value: s.value, Steps: budget - s.remaining},
		Program: int(f.program), Depth: int(s.depth), Calls: s.calls, Ticks: s.ticks, StackUsed: s.stackUsed,
	}
	runtime.KeepAlive(d)
	return result, nil
}

// Reset releases every graph borrow and invalidates execution state. It does
// not close code owned by the VM or change the caller's reference table.
func (d *Dispatch) Reset() {
	clear(d.owners[:])
	clear(d.programs[:])
	d.state = dispatchState{}
	d.begun = false
}

// Size reports the fixed dispatch allocation, excluding executable owners.
func (d *Dispatch) Size() int { return int(unsafe.Sizeof(*d)) }
