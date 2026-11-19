package vm

// regCtorCache belongs to one construction site, in regCallCache.this.
// Clearing the call caches at a turn's end releases its prepared objects.
type regCtorCache struct {
	shape *shape
	pool  *nativeCtorPool
}

type nativeCtorPool struct {
	root    *shape
	count   uint64
	objects [32]*Object
}

// nativeCtorThis makes this for the native interpreter and prepares the
// next objects for an initializer whose allocation layout has settled.
// Each object remains a separate allocation, with its own lifetime.
func (r *Runtime) nativeCtorThis(e *regCallCache, fd *funcData) *Object {
	o := r.regCtorThis(e, fd)
	if o == nil || fd.ctorKind != ctorBase || len(e.rc.initPlan) == 0 || r.jitOn() ||
		fd.ctorRoot == nil || o.shape != fd.ctorRoot || !o.shape.tracked() {
		return o
	}
	c := (*regCtorCache)(e.this.ref)
	if c == nil {
		return o
	}
	p := c.pool
	if p == nil {
		p = new(nativeCtorPool)
		c.pool = p
	}
	if p.count == 0 || p.root != o.shape {
		p.root = o.shape
		n := o.shape.roomFor(int(fd.closure.fn.ThisProps))
		for i := range p.objects {
			p.objects[i] = newShapedLiteral(p.root, ClassObject, n)
		}
		p.count = uint64(len(p.objects))
	}
	return o
}
