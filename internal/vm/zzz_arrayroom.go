package vm

import "weak"

// An empty array literal learns the length its arrays grow to, so repeated
// results need not allocate every intermediate element buffer. Two samples
// must support the size: an occasional large array must not size every small one.
// Sampling starts on the second execution: an array made by code loaded only
// once must not keep that load's object graph until a sweep. A sample is held
// weakly after a GC sweep, so a long render can still teach
// its final length without keeping its result alive. At a turn's end only
// counts remain in the closure's cache. A short result pauses sampling for
// 31 executions; a probe that grows resumes consecutive samples. Arrays
// beyond the limit keep ordinary growth, rather than a different sequence
// of capacities starting at an incomplete hint.

const arrayRoomLimit = 65536

type arrayRoom struct {
	sample         *Object
	weak           weak.Pointer[Object]
	room, previous int
	listed         bool
	wait           uint8
	acrossTurns    bool
	pc             int32
}

func arrayObserve[_ any](s *arrayRoom, a *Object) {
	n := len(a.elemsOf())
	if n > arrayRoomLimit {
		n = 0
	}
	if n < 256 {
		s.room, s.previous, s.wait = 0, 0, 31
		s.acrossTurns = false
		return
	}
	s.room = min(n, s.previous)
	s.previous = n
}

// arraySamplesLetGo is clearReturnedFrames' release of temporary samples,
// with the last for-in cursor it already released.
//
//go:noinline
func arraySamplesLetGo[_ any](r *Runtime) {
	forInLetGo[struct{}](r)
	keep := r.arraySamples[:0]
	for _, s := range r.arraySamples {
		if r.frameDepth != 0 {
			if s.sample != nil {
				// A sample may disappear before the next literal. Give
				// unused room back now, even if its final length is lost.
				if n := len(s.sample.elemsOf()); n < s.room {
					s.room = n
					if n < 256 {
						s.room = 0
					}
				}
				s.weak = weak.Make(s.sample)
				s.sample = nil
			}
			if s.weak.Value() != nil {
				keep = append(keep, s)
				continue
			}
		} else {
			a := s.sample
			if a == nil {
				a = s.weak.Value()
			}
			if a != nil {
				arrayObserve[struct{}](s, a)
				s.acrossTurns = s.room >= 256
			}
		}
		s.sample, s.weak, s.listed = nil, weak.Pointer[Object]{}, false
	}
	clear(r.arraySamples[len(keep):])
	r.arraySamples = keep
	jitLiteralArraysLetGo[struct{}](r)
}

func arrayEmptyOp[_ any](b *tbuilder, pc int) bool {
	if !b.spill() {
		return false
	}
	n := &arrayEmptyNode[struct{}]{site: b.arraySites, pc: int32(pc)}
	b.arraySites++
	b.push(n.make)
	return true
}

type arrayEmptyNode[_ any] struct {
	site int
	pc   int32
}

func arraySiteNew[_ any](cl *closureCode, site int, pc int32) *arrayRoom {
	for len(cl.arrays) <= site {
		cl.arrays = append(cl.arrays, nil)
	}
	s := &arrayRoom{wait: 1, pc: pc}
	cl.arrays[site] = s
	return s
}

func (n *arrayEmptyNode[_]) make(c *tctx) Value {
	if c.r.meter != nil {
		return Obj(c.r.newArrayFrom(nil))
	}
	var s *arrayRoom
	if n.site < len(c.arrays) {
		s = c.arrays[n.site]
	}
	if s == nil {
		s = arraySiteNew[struct{}](c.closureCode, n.site, n.pc)
	}
	track := s.wait == 0
	if track {
		previous := s.sample
		if previous == nil {
			previous = s.weak.Value()
			s.weak = weak.Pointer[Object]{}
		}
		if previous != nil {
			arrayObserve[struct{}](s, previous)
			if s.wait != 0 {
				s.sample = nil
				track = false
			}
		}
	} else {
		s.wait--
	}
	var a *Object
	if s.room == 0 {
		ao := new(arrayObject)
		ao.Object = Object{shape: rootFor(ClassArray, c.r.proto.array), class: ClassArray,
			flags: objExtensible | objArrayLengthWritable | objHasExt}
		ao.side.elems = ao.inline[:0]
		a = &ao.Object
	} else {
		a = newArrayObject(c.r.proto.array, s.room)
		a.extOf().elems = a.elemsOf()[:0]
	}
	if track {
		if !s.listed {
			c.r.arraySamples = append(c.r.arraySamples, s)
			s.listed = true
		}
		s.sample = a
	}
	return Obj(a)
}
