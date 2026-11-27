package mir

// An experiment (round r106, stage 2): internal/jit/lsra, the WebAssembly
// optimizer's allocator, run beside mir's scan on the same function, its
// result thrown away, so that -jitcompile times it against the scan's
// phases (PhaseLSRA) and LSRAStats compares what each spills.

import (
	"math/bits"
	"os"
	"sync/atomic"

	"github.com/go-quickjs/go-quickjs/internal/jit/lsra"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// lsraShadow is QJS_LSRA_SHADOW: "" off, "split" with splitting, any
// other value without.
var lsraShadow = os.Getenv("QJS_LSRA_SHADOW")

// LSRAStats sums, over the shadow's compiles, what each allocator spilled
// and what its phis' edges move, each weighed as mir's cost in a slot is.
var LSRAStats struct {
	Compiles, Errors                           atomic.Int64
	MirSpilled, LSRASpilled                    atomic.Int64
	MirSpillCost, LSRASpillCost                atomic.Int64
	MirEdgeMoves, LSRAEdgeMoves                atomic.Int64
	MirEdgeCost, LSRAEdgeCost                  atomic.Int64
	MirSlots, LSRASlots, LSRASplit, Vregs, Ins atomic.Int64
	// Ticks are the Func's build's, then lsra's phases', in Timing's
	// clock.
	Ticks [1 + lsra.NumPhases]atomic.Int64
}

// shadowIn is what allocate leaves the shadow.
type shadowIn struct {
	rec             []useRec
	xuses           []xuse
	pos, start, end []int
	count           int
}

type shadowWork struct {
	al    lsra.Allocator
	f     lsra.Func
	regs  []lsra.VReg
	insts []lsra.Inst
	phis  []lsra.Phi
	edges []int
	pargs []lsra.VReg
	// extra uses by position: head[p] 1 plus the first of extra's, next
	// links.
	head  []int32
	extra []shadowUse
	bi    []int32
	uses  []lsra.VReg
	defs  []lsra.VReg
	keep  []int
}

type shadowUse struct {
	v    lsra.VReg
	next int32
}

func grown[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n, n+n/4)
	}
	return s[:n]
}

func (c *core) lsraShadowRun(in *shadowIn) {
	var t0 int64
	if t := c.opts.Timing; t != nil {
		t0 = t.Clock()
	}
	var w *shadowWork
	if c.ws != nil {
		if c.ws.shadow == nil {
			c.ws.shadow = new(shadowWork)
		}
		w = c.ws.shadow
	} else {
		w = new(shadowWork)
	}
	f := c.f
	nv, nb := f.NumValues(), numBlocks(f)
	rec, pos, start, end := in.rec, in.pos, in.start, in.end
	masks := c.loopMasks()
	lf := &w.f
	lf.Class = grown(lf.Class, nv)
	lf.FixedSlot = grown(lf.FixedSlot, nv)
	lf.Width = nil
	lf.Clobbers = nil
	for i := range nv {
		lf.FixedSlot[i] = -1
		lf.Class[i] = lsra.NoClass
		if i < len(rec) && rec[i].def > 0 {
			lf.Class[i] = 0
			if rec[i].slot&slotFloat != 0 {
				lf.Class[i] = 1
			}
		}
	}
	alloc := func(v *ssa.Value) bool { return v.ID < nv && lf.Class[v.ID] != lsra.NoClass }
	// The uses of frame states and others mir's uses found: each value's
	// last in each block not its own (xuses), and its last in its own.
	w.head = grown(w.head, in.count+1)
	clear(w.head)
	extra := w.extra[:0]
	add := func(p int, v int) {
		extra = append(extra, shadowUse{lsra.VReg(v), w.head[p]})
		w.head[p] = int32(len(extra))
	}
	for id := range rec {
		r := &rec[id]
		if r.def <= 0 || lf.Class[id] == lsra.NoClass {
			continue
		}
		for u := r.xhead; u != 0; u = in.xuses[u-1].next {
			add(int(in.xuses[u-1].at), id)
		}
		db := r.def - 1
		if r.end > r.real && int(r.end) <= end[db] && int(r.end) > start[db] {
			add(int(r.end), id)
		}
	}
	w.extra = extra
	w.bi = grown(w.bi, nb)
	for i := range w.bi {
		w.bi[i] = -1
	}
	for i, b := range c.order {
		w.bi[b.ID] = int32(i)
	}
	lf.Blocks = grown(lf.Blocks, len(c.order))
	regs, insts, phis, edges, pargs := w.regs[:0], w.insts[:0], w.phis[:0], w.edges[:0], w.pargs[:0]
	// Room is made as it is needed, the slices taken from the arrays as
	// they are: a reallocation copies the earlier blocks' to the new array
	// and leaves theirs where they were, still valid.
	take := func(n int) []lsra.VReg {
		if cap(regs)-len(regs) < n {
			regs = make([]lsra.VReg, 0, max(2*cap(regs), n+1024))
		}
		s := regs[len(regs) : len(regs) : len(regs)+n]
		regs = regs[:len(regs)+n]
		return s
	}
	ninst := 0
	for i, b := range c.order {
		lb := &lf.Blocks[i]
		bw := min(bits.OnesCount64(masks[b.ID]), 3)
		lb.Depth = bw
		if cap(edges)-len(edges) < len(b.Preds)+len(b.Succs) {
			edges = make([]int, 0, max(2*cap(edges), 1024))
		}
		e0 := len(edges)
		keep := w.keep[:0] // predecessor indices laid out
		for k, p := range b.Preds {
			if w.bi[p.ID] >= 0 {
				edges = append(edges, int(w.bi[p.ID]))
				keep = append(keep, k)
			}
		}
		w.keep = keep
		lb.Preds = edges[e0:len(edges):len(edges)]
		e1 := len(edges)
		for _, s := range b.Succs {
			if w.bi[s.ID] >= 0 {
				edges = append(edges, int(w.bi[s.ID]))
			}
		}
		lb.Succs = edges[e1:len(edges):len(edges)]
		p0 := len(phis)
		nvals := 0
		for _, v := range b.Values {
			if v.Op == ssa.OpPhi {
				if !alloc(v) {
					continue
				}
				if cap(pargs)-len(pargs) < len(keep) {
					pargs = make([]lsra.VReg, 0, max(2*cap(pargs), 1024))
				}
				a0 := len(pargs)
				for _, k := range keep {
					a := v.Args[k]
					if alloc(a) {
						pargs = append(pargs, lsra.VReg(a.ID))
					} else {
						pargs = append(pargs, lsra.None)
					}
				}
				phis = append(phis, lsra.Phi{Def: lsra.VReg(v.ID), Args: pargs[a0:len(pargs):len(pargs)]})
				continue
			}
			nvals++
		}
		lb.Phis = phis[p0:len(phis):len(phis)]
		i0 := len(insts)
		// One instruction a value but the phis, and the block's end.
		for _, v := range b.Values {
			if v.Op == ssa.OpPhi {
				continue
			}
			uses, defs := w.uses[:0], w.defs[:0]
			fused := c.isFused(v)
			if !fused {
				for _, a := range v.Args {
					if alloc(a) {
						uses = append(uses, lsra.VReg(a.ID))
					}
				}
				for x := w.head[pos[v.ID]]; x != 0; x = extra[x-1].next {
					uses = append(uses, extra[x-1].v)
				}
				if pos[v.ID] == start[b.ID]+1 {
					for x := w.head[start[b.ID]]; x != 0; x = extra[x-1].next {
						uses = append(uses, extra[x-1].v)
					}
				}
			}
			if alloc(v) {
				defs = append(defs, lsra.VReg(v.ID))
			}
			hint := lsra.None
			if takesOperand(v) && len(v.Args) > 0 && alloc(v.Args[0]) {
				hint = lsra.VReg(v.Args[0].ID)
			}
			insts = append(insts, lsra.Inst{})
			insts[len(insts)-1].Set(take(len(uses)+len(defs)), uses, defs, nil, hint)
			w.uses, w.defs = uses, defs
		}
		uses := w.uses[:0]
		for _, v := range b.Values {
			if v.Op != ssa.OpPhi && c.isFused(v) {
				for _, a := range v.Args {
					if alloc(a) {
						uses = append(uses, lsra.VReg(a.ID))
					}
				}
			}
		}
		if b.Control != nil {
			if c.isLazy(b.Control) {
				for _, a := range b.Control.Args {
					if alloc(a) {
						uses = append(uses, lsra.VReg(a.ID))
					}
				}
			} else if alloc(b.Control) {
				uses = append(uses, lsra.VReg(b.Control.ID))
			}
		}
		for x := w.head[end[b.ID]]; x != 0; x = extra[x-1].next {
			uses = append(uses, extra[x-1].v)
		}
		if nvals == 0 {
			for x := w.head[start[b.ID]]; x != 0; x = extra[x-1].next {
				uses = append(uses, extra[x-1].v)
			}
		}
		insts = append(insts, lsra.Inst{})
		insts[len(insts)-1].Set(take(len(uses)), uses, nil, nil, lsra.None)
		w.uses = uses
		lb.Insts = insts[i0:len(insts):len(insts)]
		ninst += len(lb.Insts)
	}
	w.regs, w.insts, w.phis, w.edges, w.pargs = regs, insts, phis, edges, pargs
	cfg := lsra.Config{Classes: [][]int{c.gprs, c.fprs}, NumSlots: 1 << 30, NoSplit: lsraShadow != "split"}
	if t := c.opts.Timing; t != nil {
		last := t.Clock()
		LSRAStats.Ticks[0].Add(last - t0)
		cfg.Lap = func(p lsra.Phase) {
			now := t.Clock()
			LSRAStats.Ticks[1+p].Add(now - last)
			last = now
		}
	}
	res, err := w.al.Allocate(lf, &cfg)
	st := &LSRAStats
	st.Compiles.Add(1)
	if err != nil {
		st.Errors.Add(1)
		return
	}
	// The comparison, mir's cost in a slot of what each spills, and the
	// phis' edges whose two ends are in different places.
	var ms, ls, mc, lc, mm, lm, mec, lec, split int64
	for id := range rec {
		if rec[id].def <= 0 || lf.Class[id] == lsra.NoClass {
			continue
		}
		if l := c.locs[id]; c.hasLoc[id] && l.reg < 0 {
			ms++
			mc += int64(rec[id].cost)
		}
		if res.Reg[id] < 0 {
			ls++
			lc += int64(rec[id].cost)
		}
		if res.Split(lsra.VReg(id)) {
			split++
		}
	}
	for _, b := range c.order {
		for _, v := range b.Values {
			if v.Op != ssa.OpPhi {
				break
			}
			if !alloc(v) {
				continue
			}
			for k, a := range v.Args {
				if a == v || !alloc(a) || w.bi[b.Preds[k].ID] < 0 {
					continue
				}
				ew := int64(edgeWeights[min(bits.OnesCount64(masks[b.ID]&masks[b.Preds[k].ID]), 3)])
				lv, la := c.locs[v.ID], c.locs[a.ID]
				if (lv.reg >= 0) != (la.reg >= 0) || lv.reg >= 0 && lv.reg != la.reg || lv.reg < 0 && lv.spill != la.spill {
					mm++
					mec += ew
				}
				rv, ra := res.Reg[v.ID], res.Reg[a.ID]
				if rv != ra || rv < 0 && res.Slot[v.ID] != res.Slot[a.ID] {
					lm++
					lec += ew
				}
			}
		}
	}
	st.MirSpilled.Add(ms)
	st.LSRASpilled.Add(ls)
	st.MirSpillCost.Add(mc)
	st.LSRASpillCost.Add(lc)
	st.MirEdgeMoves.Add(mm)
	st.LSRAEdgeMoves.Add(lm)
	st.MirEdgeCost.Add(mec)
	st.LSRAEdgeCost.Add(lec)
	st.MirSlots.Add(int64(c.spills))
	st.LSRASlots.Add(int64(res.NumSlots))
	st.LSRASplit.Add(split)
	st.Ins.Add(int64(ninst))
}
