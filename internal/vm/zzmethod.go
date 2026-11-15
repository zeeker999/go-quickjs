package vm

import (
	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// recv.name(args) on a receiver read in place -- a local or this -- was a
// get_prop_this that spilled every entry beneath it, so that the receiver
// sat in its slot for the call to read again, and then a call that spilled
// the method and the arguments to their slots too: p.get() in a loop took
// 45 ns where p.x took 16 ns, half of it the spills and the nodes between
// them. A get_prop_this on such a receiver leaves a method entry instead
// (methodRead), which the call_method of it builds into one node with the
// arguments (methodCall): the receiver is read once, the method looked up,
// the arguments evaluated into their slots, in the order the bytecode
// gives them, and the call made as callNode makes it. Anything that spills
// first turns the method entry into its slot's read, and the call is then
// callNode's.
//
// They are variables, assigned closures in this file's init, laid out after
// every closure of the tree tier's (see AGENTS.md).

var (

	// leafRest is what callRest does with a call of the leaf fd's that
	// leafCall or pureCall has not answered, once it has ticked.
	leafRest func(r *Runtime, o *Object, fd *funcData, this Value, args []Value) (Value, error)
)

func init() {

}

// methodOf is what a method entry is of: the get_prop_this, and where.
type methodOf struct {
	in bytecode.Instr
	pc int
}
