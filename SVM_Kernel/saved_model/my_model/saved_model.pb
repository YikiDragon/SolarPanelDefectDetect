??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??
g
alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namealpha
`
alpha/Read/ReadVariableOpReadVariableOpalpha*
_output_shapes
:	?*
dtype0
d
biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namebias
]
bias/Read/ReadVariableOpReadVariableOpbias*
_output_shapes

:*
dtype0
l
saved_XVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	saved_X
e
saved_X/Read/ReadVariableOpReadVariableOpsaved_X* 
_output_shapes
:
??*
dtype0
k
saved_YVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name	saved_Y
d
saved_Y/Read/ReadVariableOpReadVariableOpsaved_Y*
_output_shapes
:	?*
dtype0
l
saved_KVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	saved_K
e
saved_K/Read/ReadVariableOpReadVariableOpsaved_K* 
_output_shapes
:
??*
dtype0

NoOpNoOp
?	
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?	
value?	B?	 B?	
p
SVMLayer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?
	alpha
a
bias
b
	saved_X
	X

saved_Y

Y
saved_K
K
	variables
trainable_variables
regularization_losses
	keras_api
#
0
1
	2

3
4

0
1
 
?
metrics
	variables
layer_regularization_losses
layer_metrics
trainable_variables
non_trainable_variables
regularization_losses

layers
 
DB
VARIABLE_VALUEalpha)SVMLayer/alpha/.ATTRIBUTES/VARIABLE_VALUE
B@
VARIABLE_VALUEbias(SVMLayer/bias/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEsaved_X+SVMLayer/saved_X/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEsaved_Y+SVMLayer/saved_Y/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEsaved_K+SVMLayer/saved_K/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
	2

3
4

0
1
 
?
metrics
	variables
layer_regularization_losses
layer_metrics
trainable_variables
non_trainable_variables
regularization_losses

layers
 
 
 

	0

1
2

0
 
 
 

	0

1
2
 
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1alphasaved_Ysaved_Xbias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_857414
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamealpha/Read/ReadVariableOpbias/Read/ReadVariableOpsaved_X/Read/ReadVariableOpsaved_Y/Read/ReadVariableOpsaved_K/Read/ReadVariableOpConst*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_857508
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamealphabiassaved_Xsaved_Ysaved_K*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_857533??
?
?
*__inference_svm_layer_layer_call_fn_857470

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_svm_layer_layer_call_and_return_conditional_losses_8573602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?4
?
!__inference__wrapped_model_857313
input_1/
+svm_model_svm_layer_readvariableop_resource3
/svm_model_svm_layer_mul_readvariableop_resource6
2svm_model_svm_layer_square_readvariableop_resource5
1svm_model_svm_layer_add_1_readvariableop_resource
identity??
"svm_model/svm_layer/ReadVariableOpReadVariableOp+svm_model_svm_layer_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"svm_model/svm_layer/ReadVariableOp?
&svm_model/svm_layer/mul/ReadVariableOpReadVariableOp/svm_model_svm_layer_mul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&svm_model/svm_layer/mul/ReadVariableOp?
svm_model/svm_layer/mulMul*svm_model/svm_layer/ReadVariableOp:value:0.svm_model/svm_layer/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
svm_model/svm_layer/mul?
)svm_model/svm_layer/Square/ReadVariableOpReadVariableOp2svm_model_svm_layer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)svm_model/svm_layer/Square/ReadVariableOp?
svm_model/svm_layer/SquareSquare1svm_model/svm_layer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
svm_model/svm_layer/Square?
)svm_model/svm_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2+
)svm_model/svm_layer/Sum/reduction_indices?
svm_model/svm_layer/SumSumsvm_model/svm_layer/Square:y:02svm_model/svm_layer/Sum/reduction_indices:output:0*
T0*
_output_shapes	
:?2
svm_model/svm_layer/Sum?
!svm_model/svm_layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!svm_model/svm_layer/Reshape/shape?
svm_model/svm_layer/ReshapeReshape svm_model/svm_layer/Sum:output:0*svm_model/svm_layer/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
svm_model/svm_layer/Reshape?
svm_model/svm_layer/Square_1Squareinput_1*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/Square_1?
+svm_model/svm_layer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+svm_model/svm_layer/Sum_1/reduction_indices?
svm_model/svm_layer/Sum_1Sum svm_model/svm_layer/Square_1:y:04svm_model/svm_layer/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
svm_model/svm_layer/Sum_1?
#svm_model/svm_layer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2%
#svm_model/svm_layer/Reshape_1/shape?
svm_model/svm_layer/Reshape_1Reshape"svm_model/svm_layer/Sum_1:output:0,svm_model/svm_layer/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
svm_model/svm_layer/Reshape_1?
"svm_model/svm_layer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"svm_model/svm_layer/transpose/perm?
svm_model/svm_layer/transpose	Transposeinput_1+svm_model/svm_layer/transpose/perm:output:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/transpose?
)svm_model/svm_layer/MatMul/ReadVariableOpReadVariableOp2svm_model_svm_layer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)svm_model/svm_layer/MatMul/ReadVariableOp?
svm_model/svm_layer/MatMulMatMul1svm_model/svm_layer/MatMul/ReadVariableOp:value:0!svm_model/svm_layer/transpose:y:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/MatMul
svm_model/svm_layer/Mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
svm_model/svm_layer/Mul_1/x?
svm_model/svm_layer/Mul_1Mul$svm_model/svm_layer/Mul_1/x:output:0$svm_model/svm_layer/MatMul:product:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/Mul_1?
svm_model/svm_layer/SubSub$svm_model/svm_layer/Reshape:output:0svm_model/svm_layer/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/Sub?
$svm_model/svm_layer/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$svm_model/svm_layer/transpose_1/perm?
svm_model/svm_layer/transpose_1	Transpose&svm_model/svm_layer/Reshape_1:output:0-svm_model/svm_layer/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2!
svm_model/svm_layer/transpose_1?
svm_model/svm_layer/AddAddsvm_model/svm_layer/Sub:z:0#svm_model/svm_layer/transpose_1:y:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/Add?
svm_model/svm_layer/AbsAbssvm_model/svm_layer/Add:z:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/Abs
svm_model/svm_layer/Mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
svm_model/svm_layer/Mul_2/x?
svm_model/svm_layer/Mul_2Mul$svm_model/svm_layer/Mul_2/x:output:0svm_model/svm_layer/Abs:y:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/Mul_2?
svm_model/svm_layer/ExpExpsvm_model/svm_layer/Mul_2:z:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/Exp?
svm_model/svm_layer/mul_3Mulsvm_model/svm_layer/mul:z:0svm_model/svm_layer/Exp:y:0*
T0*(
_output_shapes
:??????????2
svm_model/svm_layer/mul_3?
+svm_model/svm_layer/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2-
+svm_model/svm_layer/Sum_2/reduction_indices?
svm_model/svm_layer/Sum_2Sumsvm_model/svm_layer/mul_3:z:04svm_model/svm_layer/Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
svm_model/svm_layer/Sum_2?
(svm_model/svm_layer/Add_1/ReadVariableOpReadVariableOp1svm_model_svm_layer_add_1_readvariableop_resource*
_output_shapes

:*
dtype02*
(svm_model/svm_layer/Add_1/ReadVariableOp?
svm_model/svm_layer/Add_1Add"svm_model/svm_layer/Sum_2:output:00svm_model/svm_layer/Add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
svm_model/svm_layer/Add_1?
$svm_model/svm_layer/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$svm_model/svm_layer/transpose_2/perm?
svm_model/svm_layer/transpose_2	Transposesvm_model/svm_layer/Add_1:z:0-svm_model/svm_layer/transpose_2/perm:output:0*
T0*'
_output_shapes
:?????????2!
svm_model/svm_layer/transpose_2w
IdentityIdentity#svm_model/svm_layer/transpose_2:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::::Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?"
?
E__inference_svm_layer_layer_call_and_return_conditional_losses_857457

inputs
readvariableop_resource
mul_readvariableop_resource"
square_readvariableop_resource!
add_1_readvariableop_resource
identity?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOp?
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mul/ReadVariableOpo
mulMulReadVariableOp:value:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mul?
Square/ReadVariableOpReadVariableOpsquare_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Square/ReadVariableOpd
SquareSquareSquare/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesc
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:?2
Sumo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapem
ReshapeReshapeSum:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	?2	
ReshapeY
Square_1Squareinputs*
T0*(
_output_shapes
:??????????2

Square_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicess
Sum_1SumSquare_1:y:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape}
	Reshape_1ReshapeSum_1:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1q
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permw
	transpose	Transposeinputstranspose/perm:output:0*
T0*(
_output_shapes
:??????????2
	transpose?
MatMul/ReadVariableOpReadVariableOpsquare_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOp{
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*(
_output_shapes
:??????????2
MatMulW
Mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Mul_1/xl
Mul_1MulMul_1/x:output:0MatMul:product:0*
T0*(
_output_shapes
:??????????2
Mul_1a
SubSubReshape:output:0	Mul_1:z:0*
T0*(
_output_shapes
:??????????2
Subu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm?
transpose_1	TransposeReshape_1:output:0transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2
transpose_1^
AddAddSub:z:0transpose_1:y:0*
T0*(
_output_shapes
:??????????2
AddM
AbsAbsAdd:z:0*
T0*(
_output_shapes
:??????????2
AbsW
Mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Mul_2/xc
Mul_2MulMul_2/x:output:0Abs:y:0*
T0*(
_output_shapes
:??????????2
Mul_2O
ExpExp	Mul_2:z:0*
T0*(
_output_shapes
:??????????2
ExpZ
mul_3Mulmul:z:0Exp:y:0*
T0*(
_output_shapes
:??????????2
mul_3t
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Sum_2/reduction_indicesp
Sum_2Sum	mul_3:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_2?
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes

:*
dtype02
Add_1/ReadVariableOpu
Add_1AddSum_2:output:0Add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Add_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	Transpose	Add_1:z:0transpose_2/perm:output:0*
T0*'
_output_shapes
:?????????2
transpose_2c
IdentityIdentitytranspose_2:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_svm_model_layer_call_fn_857399
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_svm_model_layer_call_and_return_conditional_losses_8573852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
__inference__traced_save_857508
file_prefix$
 savev2_alpha_read_readvariableop#
savev2_bias_read_readvariableop&
"savev2_saved_x_read_readvariableop&
"savev2_saved_y_read_readvariableop&
"savev2_saved_k_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d8085425e6b3455ebc1be992ae8414e4/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)SVMLayer/alpha/.ATTRIBUTES/VARIABLE_VALUEB(SVMLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB+SVMLayer/saved_X/.ATTRIBUTES/VARIABLE_VALUEB+SVMLayer/saved_Y/.ATTRIBUTES/VARIABLE_VALUEB+SVMLayer/saved_K/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_alpha_read_readvariableopsavev2_bias_read_readvariableop"savev2_saved_x_read_readvariableop"savev2_saved_y_read_readvariableop"savev2_saved_k_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*O
_input_shapes>
<: :	?::
??:	?:
??: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:$ 

_output_shapes

::&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:

_output_shapes
: 
?
?
"__inference__traced_restore_857533
file_prefix
assignvariableop_alpha
assignvariableop_1_bias
assignvariableop_2_saved_x
assignvariableop_3_saved_y
assignvariableop_4_saved_k

identity_6??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)SVMLayer/alpha/.ATTRIBUTES/VARIABLE_VALUEB(SVMLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB+SVMLayer/saved_X/.ATTRIBUTES/VARIABLE_VALUEB+SVMLayer/saved_Y/.ATTRIBUTES/VARIABLE_VALUEB+SVMLayer/saved_K/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_alphaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_saved_xIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_saved_yIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_saved_kIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5?

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
E__inference_svm_model_layer_call_and_return_conditional_losses_857385
input_1
svm_layer_857375
svm_layer_857377
svm_layer_857379
svm_layer_857381
identity??!svm_layer/StatefulPartitionedCall?
!svm_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1svm_layer_857375svm_layer_857377svm_layer_857379svm_layer_857381*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_svm_layer_layer_call_and_return_conditional_losses_8573602#
!svm_layer/StatefulPartitionedCall?
IdentityIdentity*svm_layer/StatefulPartitionedCall:output:0"^svm_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2F
!svm_layer/StatefulPartitionedCall!svm_layer/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?"
?
E__inference_svm_layer_layer_call_and_return_conditional_losses_857360

inputs
readvariableop_resource
mul_readvariableop_resource"
square_readvariableop_resource!
add_1_readvariableop_resource
identity?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOp?
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
mul/ReadVariableOpo
mulMulReadVariableOp:value:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
mul?
Square/ReadVariableOpReadVariableOpsquare_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Square/ReadVariableOpd
SquareSquareSquare/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesc
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:?2
Sumo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapem
ReshapeReshapeSum:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	?2	
ReshapeY
Square_1Squareinputs*
T0*(
_output_shapes
:??????????2

Square_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicess
Sum_1SumSquare_1:y:0 Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape_1/shape}
	Reshape_1ReshapeSum_1:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1q
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permw
	transpose	Transposeinputstranspose/perm:output:0*
T0*(
_output_shapes
:??????????2
	transpose?
MatMul/ReadVariableOpReadVariableOpsquare_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOp{
MatMulMatMulMatMul/ReadVariableOp:value:0transpose:y:0*
T0*(
_output_shapes
:??????????2
MatMulW
Mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Mul_1/xl
Mul_1MulMul_1/x:output:0MatMul:product:0*
T0*(
_output_shapes
:??????????2
Mul_1a
SubSubReshape:output:0	Mul_1:z:0*
T0*(
_output_shapes
:??????????2
Subu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm?
transpose_1	TransposeReshape_1:output:0transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2
transpose_1^
AddAddSub:z:0transpose_1:y:0*
T0*(
_output_shapes
:??????????2
AddM
AbsAbsAdd:z:0*
T0*(
_output_shapes
:??????????2
AbsW
Mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Mul_2/xc
Mul_2MulMul_2/x:output:0Abs:y:0*
T0*(
_output_shapes
:??????????2
Mul_2O
ExpExp	Mul_2:z:0*
T0*(
_output_shapes
:??????????2
ExpZ
mul_3Mulmul:z:0Exp:y:0*
T0*(
_output_shapes
:??????????2
mul_3t
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Sum_2/reduction_indicesp
Sum_2Sum	mul_3:z:0 Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum_2?
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes

:*
dtype02
Add_1/ReadVariableOpu
Add_1AddSum_2:output:0Add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Add_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	Transpose	Add_1:z:0transpose_2/perm:output:0*
T0*'
_output_shapes
:?????????2
transpose_2c
IdentityIdentitytranspose_2:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_857414
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_8573132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?&
?
SVMLayer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"?
_tf_keras_model?{"class_name": "SVMModel", "name": "svm_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "SVMModel"}}
?
	alpha
a
bias
b
	saved_X
	X

saved_Y

Y
saved_K
K
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SVMLayer", "name": "svm_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 576]}}
C
0
1
	2

3
4"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
metrics
	variables
layer_regularization_losses
layer_metrics
trainable_variables
non_trainable_variables
regularization_losses

layers
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
:	?2alpha
:2bias
:
??2saved_X
:	?2saved_Y
:
??2saved_K
C
0
1
	2

3
4"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
metrics
	variables
layer_regularization_losses
layer_metrics
trainable_variables
non_trainable_variables
regularization_losses

layers
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
	0

1
2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
	0

1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
*__inference_svm_model_layer_call_fn_857399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
E__inference_svm_model_layer_call_and_return_conditional_losses_857385?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
!__inference__wrapped_model_857313?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
*__inference_svm_layer_layer_call_fn_857470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_svm_layer_layer_call_and_return_conditional_losses_857457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
3B1
$__inference_signature_wrapper_857414input_1?
!__inference__wrapped_model_857313n
	1?.
'?$
"?
input_1??????????
? "3?0
.
output_1"?
output_1??????????
$__inference_signature_wrapper_857414y
	<?9
? 
2?/
-
input_1"?
input_1??????????"3?0
.
output_1"?
output_1??????????
E__inference_svm_layer_layer_call_and_return_conditional_losses_857457_
	0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
*__inference_svm_layer_layer_call_fn_857470R
	0?-
&?#
!?
inputs??????????
? "???????????
E__inference_svm_model_layer_call_and_return_conditional_losses_857385`
	1?.
'?$
"?
input_1??????????
? "%?"
?
0?????????
? ?
*__inference_svm_model_layer_call_fn_857399S
	1?.
'?$
"?
input_1??????????
? "??????????