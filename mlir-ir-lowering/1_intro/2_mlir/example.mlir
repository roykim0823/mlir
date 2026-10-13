// loop_add: sum the integers 0..9 using a structured (scf) for-loop.
// Returns `index` (a platform-sized integer, like size_t).
func.func @loop_add() -> (index) {
  // `index` dialect: loop bounds and the running total are index-typed.
  %init = index.constant 0   // initial accumulator value
  %lb = index.constant 0     // loop lower bound (inclusive)
  %ub = index.constant 10    // loop upper bound (exclusive)
  %step = index.constant 1   // loop step

  // scf.for is a *structured* loop. `iter_args` threads a loop-carried value
  // (%acc) through each iteration; whatever we scf.yield becomes %acc next
  // time, and the final value is bound to %sum.
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (index) {
    %sum_next = arith.addi %acc, %iv : index   // arith dialect: acc + iv
    scf.yield %sum_next : index                // carry sum_next into next iter
  }
  return %sum : index
}

// main: call loop_add and return the result as a C-style i32 exit code.
func.func @main() -> i32 {
  %out = call @loop_add() : () -> index
  // index is i64 here; narrow it to i32 so main can return a normal exit code.
  %out_i32 = arith.index_cast %out : index to i32
  return %out_i32 : i32
}

// In C this is equivalent to:
//int loop_add(int lb, int ub, int step) {
//  int sum_0 = 0;
//  int sum = sum_0;
//  for (int iv = lb; iv < ub; iv += step) {
//    int sum_next = sum + iv;
//    sum = sum_next;
//  }
//  return sum;
//}

//int main() {
//  int out = loop_add(0, 10, 1);
//  return out;
//}