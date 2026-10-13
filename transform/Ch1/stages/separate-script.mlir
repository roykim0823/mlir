// A transform script kept apart from the payload: no debug-bind-trailing-args,
// the handles come from matching inside the payload root instead.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.op<"linalg.matmul">
    %elemwise = transform.structured.match ops{["linalg.elemwise_binary"]} in %root
        : (!transform.any_op) -> !transform.op<"linalg.elemwise_binary">
    transform.print %matmul {name = "matched matmul"} : !transform.op<"linalg.matmul">
    transform.print %elemwise {name = "matched elemwise_binary"}
        : !transform.op<"linalg.elemwise_binary">
    %tiled, %loop = transform.structured.tile_using_forall %matmul tile_sizes [4, 32]
        : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)
    transform.print %loop {name = "loop after tiling"} : !transform.any_op
    transform.yield
  }
}
