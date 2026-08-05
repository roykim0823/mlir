# The `Broadcastable` Trait

> **Section:** Core IR · document 6 of 7  
> **Upstream:** [https://mlir.llvm.org/docs/Traits/Broadcastable/](https://mlir.llvm.org/docs/Traits/Broadcastable/) · source [`mlir/docs/Traits/Broadcastable.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Traits/Broadcastable.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

A deep dive on a single trait, and the best available worked example of what a substantial trait
looks like: `ResultsBroadcastableShape` encodes NumPy-style broadcasting so that elementwise
operations with mismatched but compatible shapes verify correctly and infer their result shape.

Read it for two reasons. If you work with machine-learning frontends you need these rules precisely,
because almost every framework has a subtly different notion of broadcasting and the mismatches
surface as shape errors deep in a pipeline. And if you are designing your own trait, this is the
model to imitate: a formal rule statement, then the verification and inference it implies.

**Read first**

- [Traits](05-traits.md)

**What you should be able to do after this page**

- State exactly when two shapes are broadcast-compatible, including dynamic dimensions.
- Predict the inferred result shape of a broadcasting op.
- Use this as a template when designing a trait of your own.

---

## Upstream documentation

## Description

The `Broadcastable` trait enforces the following properties on an operation:

- The operation has at least one input operand.

- The operation has exactly one result.

- All input operands and result are of type `tensor` or `vector`.

- A shape inference mechanism is able to compute the result shape solely based on input operand shapes.

- Input operands have broadcast-compatible shapes, according to the verification rules presented below.

- The operation's result shape is compatible with &mdash;though not necessarily identical to&mdash; the shape inferred from its input operands, according to the verification rules presented below.


## Dimension inference

Given an operation with two input operands, the size of dimension `i` of its result can be inferred from dimension `i` of the operands according to the table below. Here, `dim0` and `dim1` represent dimension `i` of the input operands in an interchangeable order, while `inferredDim` represents the inferred size for dimension `i` of the operation result. Dimensions are classified in three categories: dynamic ("?"), static equal to 1 ("1"), and static greater than 1 (">1").


| `dim0` | `dim1` | `inferredDim` | Notes |
| -------- | -------- | ------------- | ----- |
| ? | ? | ? | If `RuntimeSize(dim0)` is 1, dimension `dim0` is broadcast to `RuntimeSize(dim1)`. If `RuntimeSize(dim1)` is 1, dimension `dim1` is broadcast to `RuntimeSize(dim0)`. The operation produces undefined behavior if both runtime sizes are greater than 1 and not equal. |
| ? | 1 | ? | Dimension `dim1` is broadcast to `RuntimeSize(dim0)`. |
| ? | >1 | `dim1` | If `RuntimeSize(dim0)` is 1, `dim0` is broadcast to `dim1`. The operation produces undefined behavior if `RuntimeSize(dim0)` is greater than 1 and not equal to `dim1`. |
| 1 | 1 | 1 | |
| 1 | >1 | `dim1` | Dimension `dim0` is broadcast to `dim1`. |
| >1 | >1 | `dim0` | The operation verifier produces a compile-time error if `dim0` != `dim1`. |


The following pseudo-function is a formal representation of the dimension inference process:

```python
InferDim(dim0, dim1):
	switch (dim0, dim1):
		case (?, ?):
		case (?, 1):
		case (1, 1):
		case (>1, ?):
		case (>1, 1):
			return dim0
		case (?, >1):
		case (1, ?):
		case (1, >1):
			return dim1
		case (>1, >1):
			ERROR_IF(dim0 != dim1)
			return dim0
```

## Shape inference

The shape inference process begins by correcting rank differences in input operands. A shape is expanded by adding additional dimensions of size 1 on its left until the desired rank is reached, as shown here:

```python
ExpandRank(shape, rank):
	while len(shape) < rank:
		shape.prepend(1)
```
		
Given the shapes of two ranked input operands, the result's shape is inferred by equalizing input ranks and inferring individual dimensions, as shown here:

```python
InferShape(shape0, shape1):

  # Equalize ranks
  rank = max(GetRank(shape0), GetRank(shape1))
  ExpandRank(shape0, rank)
  ExpandRank(shape1, rank)
	
  # Infer shape
  inferredShape = []
  for (dim0, dim1) in zip(shape0, shape1):
    inferredDim = InferDim(dim0, dim1)
    inferredShape.append(inferredDim)
  return inferredShape
```
	
The result shape for an operation with an arbitrary number of input operands is then inferred by discarding unranked operands, applying shape inference on the first ranked operand pair, and updating the inferred shape with each additional ranked operand. If the operation has no ranked operands, the result shape cannot be inferred. If the operation has exactly one ranked operand, its shape is directly provided as the inferred result shape. Formally:

```python
InferResultShape(op):

	# Filter ranked operands
	rankedOperands = filter(op.operands, IsRanked)
	if len(rankedOperands) == 0:
		return None
	
	# Infer result shape
	inferredShape = GetShape(rankedOperands[0])
	for operand in rankedOperands[1:]:
		inferredShape = InferShape(inferredShape, GetShape(operand))
	return inferredShape
```

## Verification

The legality of an operation with the `Broadcastable` trait is verified by first running the shape inference process. If a failure occurs during shape inference, it is concluded that input operands are not broadcast-compatible, and verification fails. If shape inference succeeds, verification continues.

If either the result is unranked or all input operands are unranked, no further verification steps are needed, and the process ends here successfully. If, on the contrary, both the result and at least one input operand are ranked, verification continues by checking for a matching rank between the previously inferred shape and the result.

Once a rank match is guaranteed, each dimension of the inferred shape is compared with the corresponding dimension of the actual result shape according to the following table table:


| `inferredDim` | `actualDim` | Verification outcome |
| ------------- | ----------- | -------------------- |
| ? | ? | **OK** |
| ? | static | **OK** <br> A failure to guarantee that the runtime dimension size of the result is equal to `actualDim` causes undefined behavior. While unusual, this implicit dynamic-to-static cast is convenient in certain scenarios, such as an intermediate state of a shape inference pass. Ultimately, a static dimension in the result implies that all input dimension sizes are also known at compile time and may therefore become static as well, preferably. |
| static | ? | **OK** <br> The actual result dimension may be dynamic even when a static size can be inferred at compile time. The programmer may choose to relax the specificity of the result dimension for forward compatibility of the result type. |
| static | static | **OK if equal** <br> When both the inferred and actual dimensions are static, they must be set to the same size. |


The full verification process can be formally specified as follows:

```python
Verify(op):

	# Run shape inference
	inferredShape = InferResultShape(op.operands)

	# Done if result is unranked or all operands are unranked
	if not IsRanked(op.result) or inferredShape is None:
		return
	
	# Rank must match
	actualShape = GetShape(op.result):
	ERROR_IF(len(inferredShape) != len(actualShape))
	
	# Verify
	for (inferredDim, actualDim) in zip(inferredShape, actualShape):
		ERROR_IF(IsStatic(actualDim) and inferredDim != actualDim)
```
		
## Examples

The following are correct uses of broadcastable ops:

```mlir
// Exact match of static sizes.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<1x2xi32) -> tensor<1x2xi32>

// Dynamic sizes match. The programmer must guarantee that the runtime sizes of
// %arg0 and %arg1 are equal at runtime.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32) -> tensor<?xi32>

// The shape of %arg0 is broadcast from tensor<1xi32> to tensor<4xi32>.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<1xi32>, tensor<4xi32) -> tensor<4xi32>

// The shape of %result is inferred as tensor<4xi32>, while the actual result
// type is tensor<?xi32>. The inferred shape is compatible with the actual shape.
%result = "test.broadcastable"(%arg0) : (tensor<4xi32) -> tensor<?xi32>

// The shape of %arg0 is first expanded to tensor<1x1x4xi32> and then broadcast
// to tensor<2x3x4xi32>.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<4xi32>, tensor<2x3x4xi32) -> tensor<2x3x4xi32>

// Input and results tensors have different element types (i1, i32, i64). The
// 'Broadcastable' trait has no restrictions on element types.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<2xi1>, tensor<2xi32) -> tensor<2xi64>

// No result shape verification is needed when the result is unranked.
%result = "test.broadcastable"(%arg0) : (tensor<2xi32>) -> tensor<*xi32>

// No result shape verification needed when all inputs are unranked.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<2xi32>
```


The following are incorrect uses of broadcastable ops:

```mlir
// Dimension 0 of input operands is static but not equal.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<3xi32>, tensor<2xi32) -> tensor<?xi32>

// The inferred result shape is tensor<3xi32>, but the actual result shape is
// tensor<1x3xi32>. Inferred and actual shapes differ in rank.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<3xi32>, tensor<3xi32) -> tensor<1x3xi32>

// The inferred result shape is tensor<?xi32>, but the actual shape is
// tensor<4xi32>. The inferred shape is not compatible with the actual shape.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32) -> tensor<4xi32>

// The inferred result shape is tensor<2xi32>, but the actual result shape is
// tensor<4xi32>, which is not compatible.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32) -> tensor<4xi32>

// The inferred result shape is tensor<1xi32>, but the actual result shape is
// tensor<4xi32>. Broadcast semantics are not applicable for results.
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32) -> tensor<4xi32>
```

---

## Deeper notes

### The rule in one paragraph

Align the two shapes at their trailing dimensions. For each aligned pair, the dimensions are
compatible if they are equal, or if one of them is 1 (which is stretched). Missing leading
dimensions on the shorter shape behave as 1. The result takes the larger of each pair.

### Where dynamic dimensions complicate it

A dynamic dimension `?` is compatible with anything, because at compile time you cannot rule out
either that it equals the other dimension or that it equals 1. The result dimension is then also
dynamic unless the static partner is 1 — and even then only if you are prepared to assert that the
dynamic dimension is not 1. This asymmetry is precisely where framework semantics diverge:

- some frontends treat a static 1 against a dynamic `?` as producing `?`;
- some treat it as producing the dynamic value, i.e. assume the 1 is the one being stretched;
- some insert a runtime check.

If you import from more than one framework, make the choice explicit at import time rather than
relying on a default, or you will get shape mismatches that only appear on particular input shapes
at run time.

### Lesson for trait design

The structure this page follows is worth copying:

1. Define the property formally, including edge cases, before any code.
2. Derive the verifier from the definition.
3. Derive shape inference from the same definition, so verification and inference cannot drift.
4. Enumerate the cases you deliberately reject.

Traits that skip step 1 and start from an implementation tend to acquire inconsistencies between
what the verifier accepts and what inference computes — which produces IR that passes verification
and then fails to lower.

### Relationship to shape inference

This trait is one specific inference rule. The general machinery — the interfaces, when to run
refinement, how it interacts with canonicalization — is covered in
[Shape Inference](../02-defining-dialects/07-shape-inference.md). Read that page after this one if
you are building a frontend.


---

[← Traits](05-traits.md) · [Index](../README.md) · [Interfaces →](07-interfaces.md)
