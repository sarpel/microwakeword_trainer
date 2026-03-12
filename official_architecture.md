════════════════════════════════════════════════════════════════════════════════════════════════════
TFLite General-Purpose Architecture Analyzer
════════════════════════════════════════════════════════════════════════════════════════════════════
Model:         C:\Users\Sarpel\Desktop\analyzer\okay_nabu.tflite
Model size:    60264 bytes (58.85 KB)
Subgraphs:     2
Quantization:  yes
Variables:     yes
Unique ops:    13 (all subgraphs via tflite package)

ESPHome microWakeWord Architecture Compliance Check
───────────────────────────────────────────────────
Rule                       │ Status │ Message
───────────────────────────┼────────┼──────────────────────────────────────────────────────────────────
I/O Dtypes (Art. II)       │ PASS   │ Input is int8, Output is uint8.
Allowed Ops (Art. IV)      │ PASS   │ All operations are within the registered 20 ESPHome op resolvers.
Dual Subgraph (Art. V)     │ PASS   │ Model has exactly 2 subgraphs (Main + Init).
Var Quantization (Art. IX) │ PASS   │ All state variable data payloads are correctly quantized to int8.

Model Inputs
────────────
Index │ Name                          │ Shape      │ Dtype │ Quantization
──────┼───────────────────────────────┼────────────┼───────┼────────────────
0     │ serving_default_input_audio:0 │ [1, 3, 40] │ int8  │ scales=1, dim=0

Model Outputs
─────────────
Index │ Name                      │ Shape  │ Dtype │ Quantization
──────┼───────────────────────────┼────────┼───────┼────────────────
93    │ StatefulPartitionedCall:0 │ [1, 1] │ uint8 │ scales=1, dim=0

Subgraph Overview
─────────────────
Subgraph │ Tensors │ Ops │ Inputs │ Outputs │ Memory (bytes)
─────────┼─────────┼─────┼────────┼─────────┼───────────────
0        │ 95      │ 55  │ [0]    │ [93]    │ 41771
1        │ 12      │ 12  │ [0]    │ []      │ 3520

Per-Subgraph Tensor Details
───────────────────────────

Subgraph 0
Index │ Role         │ Name                                 │ Shape          │ Dtype  │ Mem(B) │ Quant
──────┼──────────────┼──────────────────────────────────────┼────────────────┼────────┼────────┼──────────────────────────
0     │ input        │ serving_default_input_audio:0        │ [1, 3, 40]     │ int8   │ 120    │ [Q: s=0.101961, z=-128]
1     │ intermediate │ model/tf.expand_dims/ExpandDims      │ [4]            │ int32  │ 16     │
2     │ intermediate │ model/stream/strided_slice           │ [4]            │ int32  │ 16     │
3     │ intermediate │ model/stream/strided_slice1          │ [4]            │ int32  │ 16     │
4     │ intermediate │ model/strided_keep/strided_slice/st… │ [4]            │ int32  │ 16     │
5     │ intermediate │ model/stream_1/strided_slice         │ [4]            │ int32  │ 16     │
6     │ intermediate │ model/stream_1/strided_slice1        │ [4]            │ int32  │ 16     │
7     │ intermediate │ model/stream_2/strided_slice         │ [4]            │ int32  │ 16     │
8     │ intermediate │ model/stream_5/strided_slice         │ [4]            │ int32  │ 16     │
9     │ intermediate │ model/tf.split/Const                 │ [2]            │ int32  │ 8      │
10    │ intermediate │ model/tf.concat/concat/axis          │ []             │ int32  │ 4      │
11    │ intermediate │ model/strided_keep/strided_slice/st… │ [4]            │ int32  │ 16     │
12    │ intermediate │ model/strided_keep/strided_slice/st… │ [4]            │ int32  │ 16     │
13    │ intermediate │ model/strided_keep_1/strided_slice/… │ [4]            │ int32  │ 16     │
14    │ intermediate │ model/stream_3/strided_slice         │ [4]            │ int32  │ 16     │
15    │ intermediate │ model/strided_keep_2/strided_slice/… │ [4]            │ int32  │ 16     │
16    │ intermediate │ model/strided_keep_3/strided_slice/… │ [4]            │ int32  │ 16     │
17    │ intermediate │ model/stream_4/strided_slice         │ [4]            │ int32  │ 16     │
18    │ intermediate │ model/stream_5/strided_slice1        │ [4]            │ int32  │ 16     │
19    │ intermediate │ model/flatten/Const                  │ [2]            │ int32  │ 8      │
20    │ intermediate │ model/dense/BiasAdd/ReadVariableOp   │ [1]            │ int32  │ 4      │ [Q: s=9.03495e-05, z=0]
21    │ intermediate │ model/dense/MatMul                   │ [1, 384]       │ int8   │ 384    │ [Q: s=0.00343903, z=0]
22    │ intermediate │ model/batch_normalization_3/FusedBa… │ [64]           │ int32  │ 256    │ [Q: per-axis n=64, dim=0]
23    │ intermediate │ model/conv2d_4/Conv2D                │ [64, 1, 1, 64] │ int8   │ 4096   │ [Q: per-axis n=64, dim=0]
24    │ intermediate │ model/depthwise_conv2d_5/BiasAdd/Re… │ [64]           │ int32  │ 256    │ [Q: per-axis n=64, dim=0]
25    │ intermediate │ model/depthwise_conv2d_5/depthwise   │ [1, 23, 1, 64] │ int8   │ 1472   │ [Q: per-axis n=64, dim=3]
26    │ intermediate │ model/batch_normalization_2/FusedBa… │ [64]           │ int32  │ 256    │ [Q: per-axis n=64, dim=0]
27    │ intermediate │ model/conv2d_3/Conv2D                │ [64, 1, 1, 64] │ int8   │ 4096   │ [Q: per-axis n=64, dim=0]
28    │ intermediate │ model/depthwise_conv2d_4/BiasAdd/Re… │ [32]           │ int32  │ 128    │ [Q: per-axis n=32, dim=0]
29    │ intermediate │ model/depthwise_conv2d_4/depthwise1  │ [1, 15, 1, 32] │ int8   │ 480    │ [Q: per-axis n=32, dim=3]
30    │ intermediate │ model/depthwise_conv2d_3/BiasAdd/Re… │ [32]           │ int32  │ 128    │ [Q: per-axis n=32, dim=0]
31    │ intermediate │ model/depthwise_conv2d_3/depthwise   │ [1, 9, 1, 32]  │ int8   │ 288    │ [Q: per-axis n=32, dim=3]
32    │ intermediate │ model/batch_normalization_1/FusedBa… │ [64]           │ int32  │ 256    │ [Q: per-axis n=64, dim=0]
33    │ intermediate │ model/conv2d_2/Conv2D                │ [64, 1, 1, 64] │ int8   │ 4096   │ [Q: per-axis n=64, dim=0]
34    │ intermediate │ model/depthwise_conv2d_2/BiasAdd/Re… │ [32]           │ int32  │ 128    │ [Q: per-axis n=32, dim=0]
35    │ intermediate │ model/depthwise_conv2d_2/depthwise   │ [1, 11, 1, 32] │ int8   │ 352    │ [Q: per-axis n=32, dim=3]
36    │ intermediate │ model/depthwise_conv2d_1/BiasAdd/Re… │ [32]           │ int32  │ 128    │ [Q: per-axis n=32, dim=0]
37    │ intermediate │ model/depthwise_conv2d_1/depthwise   │ [1, 7, 1, 32]  │ int8   │ 224    │ [Q: per-axis n=32, dim=3]
38    │ intermediate │ model/batch_normalization/FusedBatc… │ [64]           │ int32  │ 256    │ [Q: per-axis n=64, dim=0]
39    │ intermediate │ model/conv2d_1/Conv2D                │ [64, 1, 1, 32] │ int8   │ 2048   │ [Q: per-axis n=64, dim=0]
40    │ intermediate │ model/depthwise_conv2d/BiasAdd/Read… │ [32]           │ int32  │ 128    │ [Q: per-axis n=32, dim=0]
41    │ intermediate │ model/depthwise_conv2d/depthwise     │ [1, 5, 1, 32]  │ int8   │ 160    │ [Q: per-axis n=32, dim=3]
42    │ intermediate │ model/depthwise_conv2d_4/depthwise   │ [32]           │ int32  │ 128    │ [Q: per-axis n=32, dim=0]
43    │ intermediate │ model/stream/conv2d/Conv2D           │ [32, 5, 1, 40] │ int8   │ 6400   │ [Q: per-axis n=32, dim=0]
44    │ variable     │ stream/states                        │ []             │ object │ 8      │
45    │ variable     │ stream_1/states                      │ []             │ object │ 8      │
46    │ variable     │ stream_2/states                      │ []             │ object │ 8      │
47    │ variable     │ stream_3/states                      │ []             │ object │ 8      │
48    │ variable     │ stream_4/states                      │ []             │ object │ 8      │
49    │ variable     │ stream_5/states                      │ []             │ object │ 8      │
50    │ intermediate │ model/tf.expand_dims/ExpandDims1     │ [1, 3, 1, 40]  │ int8   │ 120    │ [Q: s=0.101961, z=-128]
51    │ variable     │ model/stream_1/concat/ReadVariableOp │ [1, 4, 1, 32]  │ int8   │ 128    │ [Q: s=1.27439, z=-128]
52    │ variable     │ model/stream_2/concat/ReadVariableOp │ [1, 10, 1, 64] │ int8   │ 640    │ [Q: s=0.0345457, z=-128]
53    │ variable     │ model/stream_3/concat/ReadVariableOp │ [1, 14, 1, 64] │ int8   │ 896    │ [Q: s=0.0408709, z=-128]
54    │ variable     │ model/stream_4/concat/ReadVariableOp │ [1, 22, 1, 64] │ int8   │ 1408   │ [Q: s=0.0319873, z=-128]
55    │ variable     │ model/stream_5/concat/ReadVariableOp │ [1, 5, 1, 64]  │ int8   │ 320    │ [Q: s=0.0262718, z=-128]
56    │ variable     │ model/stream/concat/ReadVariableOp   │ [1, 2, 1, 40]  │ int8   │ 80     │ [Q: s=0.101961, z=-128]
57    │ intermediate │ model/stream/concat                  │ [1, 5, 1, 40]  │ int8   │ 200    │ [Q: s=0.101961, z=-128]
58    │ variable     │ model/stream/strided_slice2          │ [1, 2, 1, 40]  │ int8   │ 80     │ [Q: s=0.101961, z=-128]
59    │ intermediate │ model/activation/Relu;model/stream/… │ [1, 1, 1, 32]  │ int8   │ 32     │ [Q: s=1.27439, z=-128]
60    │ intermediate │ model/stream_1/concat                │ [1, 5, 1, 32]  │ int8   │ 160    │ [Q: s=1.27439, z=-128]
61    │ variable     │ model/stream_1/strided_slice2        │ [1, 4, 1, 32]  │ int8   │ 128    │ [Q: s=1.27439, z=-128]
62    │ intermediate │ model/depthwise_conv2d/BiasAdd;mode… │ [1, 1, 1, 32]  │ int8   │ 32     │ [Q: s=1.10932, z=-28]
63    │ intermediate │ model/activation_1/Relu;model/batch… │ [1, 1, 1, 64]  │ int8   │ 64     │ [Q: s=0.0345457, z=-128]
64    │ intermediate │ model/stream_2/concat                │ [1, 11, 1, 64] │ int8   │ 704    │ [Q: s=0.0345457, z=-128]
65    │ variable     │ model/stream_2/strided_slice1        │ [1, 10, 1, 64] │ int8   │ 640    │ [Q: s=0.0345457, z=-128]
66    │ intermediate │ model/tf.split/split;model/tf.split… │ [1, 11, 1, 32] │ int8   │ 352    │ [Q: s=0.0345457, z=-128]
67    │ intermediate │ model/tf.split/split;model/tf.split… │ [1, 11, 1, 32] │ int8   │ 352    │ [Q: s=0.0345457, z=-128]
68    │ intermediate │ model/strided_keep/strided_slice     │ [1, 7, 1, 32]  │ int8   │ 224    │ [Q: s=0.0345457, z=-128]
69    │ intermediate │ model/depthwise_conv2d_1/BiasAdd;mo… │ [1, 1, 1, 32]  │ int8   │ 32     │ [Q: s=0.0474272, z=-18]
70    │ intermediate │ model/strided_keep_1/strided_slice   │ [1, 11, 1, 32] │ int8   │ 352    │ [Q: s=0.0345457, z=-128]
71    │ intermediate │ model/depthwise_conv2d_2/BiasAdd;mo… │ [1, 1, 1, 32]  │ int8   │ 32     │ [Q: s=0.0474272, z=-18]
72    │ intermediate │ model/tf.concat/concat               │ [1, 1, 1, 64]  │ int8   │ 64     │ [Q: s=0.0474272, z=-18]
73    │ intermediate │ model/activation_2/Relu;model/batch… │ [1, 1, 1, 64]  │ int8   │ 64     │ [Q: s=0.0408709, z=-128]
74    │ intermediate │ model/stream_3/concat                │ [1, 15, 1, 64] │ int8   │ 960    │ [Q: s=0.0408709, z=-128]
75    │ variable     │ model/stream_3/strided_slice1        │ [1, 14, 1, 64] │ int8   │ 896    │ [Q: s=0.0408709, z=-128]
76    │ intermediate │ model/tf.split_1/split;model/tf.spl… │ [1, 15, 1, 32] │ int8   │ 480    │ [Q: s=0.0408709, z=-128]
77    │ intermediate │ model/tf.split_1/split;model/tf.spl… │ [1, 15, 1, 32] │ int8   │ 480    │ [Q: s=0.0408709, z=-128]
78    │ intermediate │ model/strided_keep_2/strided_slice   │ [1, 9, 1, 32]  │ int8   │ 288    │ [Q: s=0.0408709, z=-128]
79    │ intermediate │ model/depthwise_conv2d_3/BiasAdd;mo… │ [1, 1, 1, 32]  │ int8   │ 32     │ [Q: s=0.0715421, z=4]
80    │ intermediate │ model/strided_keep_3/strided_slice   │ [1, 15, 1, 32] │ int8   │ 480    │ [Q: s=0.0408709, z=-128]
81    │ intermediate │ model/depthwise_conv2d_4/BiasAdd;mo… │ [1, 1, 1, 32]  │ int8   │ 32     │ [Q: s=0.0715421, z=4]
82    │ intermediate │ model/tf.concat_1/concat             │ [1, 1, 1, 64]  │ int8   │ 64     │ [Q: s=0.0715421, z=4]
83    │ intermediate │ model/activation_3/Relu;model/batch… │ [1, 1, 1, 64]  │ int8   │ 64     │ [Q: s=0.0319873, z=-128]
84    │ intermediate │ model/stream_4/concat                │ [1, 23, 1, 64] │ int8   │ 1472   │ [Q: s=0.0319873, z=-128]
85    │ variable     │ model/stream_4/strided_slice1        │ [1, 22, 1, 64] │ int8   │ 1408   │ [Q: s=0.0319873, z=-128]
86    │ intermediate │ model/depthwise_conv2d_5/BiasAdd;mo… │ [1, 1, 1, 64]  │ int8   │ 64     │ [Q: s=0.113524, z=3]
87    │ intermediate │ model/activation_4/Relu;model/batch… │ [1, 1, 1, 64]  │ int8   │ 64     │ [Q: s=0.0262718, z=-128]
88    │ intermediate │ model/stream_5/concat                │ [1, 6, 1, 64]  │ int8   │ 384    │ [Q: s=0.0262718, z=-128]
89    │ variable     │ model/stream_5/strided_slice2        │ [1, 5, 1, 64]  │ int8   │ 320    │ [Q: s=0.0262718, z=-128]
90    │ intermediate │ model/flatten/Reshape                │ [1, 384]       │ int8   │ 384    │ [Q: s=0.0262718, z=-128]
91    │ intermediate │ model/dense/MatMul;model/dense/Bias… │ [1, 1]         │ int8   │ 1      │ [Q: s=0.152942, z=46]
92    │ intermediate │ StatefulPartitionedCall:01           │ [1, 1]         │ int8   │ 1      │ [Q: s=0.00390625, z=-128]
93    │ output       │ StatefulPartitionedCall:0            │ [1, 1]         │ uint8  │ 1      │ [Q: s=0.00390625, z=0]
94    │ intermediate │                                      │ [1, 1, 1, 200] │ int8   │ 200    │

Subgraph 1
Index │ Role         │ Name               │ Shape          │ Dtype  │ Mem(B) │ Quant
──────┼──────────────┼────────────────────┼────────────────┼────────┼────────┼─────────────────────────
0     │ input        │ stream_5/states1   │ []             │ object │ 8      │
1     │ intermediate │ tfl.pseudo_qconst  │ [1, 5, 1, 64]  │ int8   │ 320    │ [Q: s=0.0262718, z=-128]
2     │ variable     │ stream_4/states1   │ []             │ object │ 8      │
3     │ intermediate │ tfl.pseudo_qconst1 │ [1, 22, 1, 64] │ int8   │ 1408   │ [Q: s=0.0319873, z=-128]
4     │ variable     │ stream_3/states1   │ []             │ object │ 8      │
5     │ intermediate │ tfl.pseudo_qconst2 │ [1, 14, 1, 64] │ int8   │ 896    │ [Q: s=0.0408709, z=-128]
6     │ variable     │ stream_2/states1   │ []             │ object │ 8      │
7     │ intermediate │ tfl.pseudo_qconst3 │ [1, 10, 1, 64] │ int8   │ 640    │ [Q: s=0.0345457, z=-128]
8     │ variable     │ stream_1/states1   │ []             │ object │ 8      │
9     │ intermediate │ tfl.pseudo_qconst4 │ [1, 4, 1, 32]  │ int8   │ 128    │ [Q: s=1.27439, z=-128]
10    │ variable     │ stream/states1     │ []             │ object │ 8      │
11    │ intermediate │ tfl.pseudo_qconst5 │ [1, 2, 1, 40]  │ int8   │ 80     │ [Q: s=0.101961, z=-128]

Operations by Subgraph
──────────────────────

  ── Subgraph 0 ──────────────────────────────────
  Note: Subgraph 0 operations from flatbuffer parser (tflite package).
Index │ Op                │ Inputs     │ Outputs
──────┼───────────────────┼────────────┼────────
0     │ CALL_ONCE         │ —          │ —
1     │ VAR_HANDLE        │ —          │ 44
2     │ VAR_HANDLE        │ —          │ 45
3     │ VAR_HANDLE        │ —          │ 46
4     │ VAR_HANDLE        │ —          │ 47
5     │ VAR_HANDLE        │ —          │ 48
6     │ VAR_HANDLE        │ —          │ 49
7     │ RESHAPE           │ 0,1        │ 50
8     │ READ_VARIABLE     │ 45         │ 51
9     │ READ_VARIABLE     │ 46         │ 52
10    │ READ_VARIABLE     │ 47         │ 53
11    │ READ_VARIABLE     │ 48         │ 54
12    │ READ_VARIABLE     │ 49         │ 55
13    │ READ_VARIABLE     │ 44         │ 56
14    │ CONCATENATION     │ 56,50      │ 57
15    │ STRIDED_SLICE     │ 57,2,3,4   │ 58
16    │ ASSIGN_VARIABLE   │ 44,58      │ —
17    │ CONV_2D           │ 57,43,42   │ 59
18    │ CONCATENATION     │ 51,59      │ 60
19    │ STRIDED_SLICE     │ 60,5,6,4   │ 61
20    │ ASSIGN_VARIABLE   │ 45,61      │ —
21    │ DEPTHWISE_CONV_2D │ 60,41,40   │ 62
22    │ CONV_2D           │ 62,39,38   │ 63
23    │ CONCATENATION     │ 52,63      │ 64
24    │ STRIDED_SLICE     │ 64,7,8,4   │ 65
25    │ ASSIGN_VARIABLE   │ 46,65      │ —
26    │ SPLIT_V           │ 64,9,10    │ 66,67
27    │ STRIDED_SLICE     │ 66,11,12,4 │ 68
28    │ DEPTHWISE_CONV_2D │ 68,37,36   │ 69
29    │ STRIDED_SLICE     │ 67,13,12,4 │ 70
30    │ DEPTHWISE_CONV_2D │ 70,35,34   │ 71
31    │ CONCATENATION     │ 69,71      │ 72
32    │ CONV_2D           │ 72,33,32   │ 73
33    │ CONCATENATION     │ 53,73      │ 74
34    │ STRIDED_SLICE     │ 74,14,8,4  │ 75
35    │ ASSIGN_VARIABLE   │ 47,75      │ —
36    │ SPLIT_V           │ 74,9,10    │ 76,77
37    │ STRIDED_SLICE     │ 76,15,12,4 │ 78
38    │ DEPTHWISE_CONV_2D │ 78,31,30   │ 79
39    │ STRIDED_SLICE     │ 77,16,12,4 │ 80
40    │ DEPTHWISE_CONV_2D │ 80,29,28   │ 81
41    │ CONCATENATION     │ 79,81      │ 82
42    │ CONV_2D           │ 82,27,26   │ 83
43    │ CONCATENATION     │ 54,83      │ 84
44    │ STRIDED_SLICE     │ 84,17,8,4  │ 85
45    │ ASSIGN_VARIABLE   │ 48,85      │ —
46    │ DEPTHWISE_CONV_2D │ 84,25,24   │ 86
47    │ CONV_2D           │ 86,23,22   │ 87
48    │ CONCATENATION     │ 55,87      │ 88
49    │ STRIDED_SLICE     │ 88,18,8,4  │ 89
50    │ ASSIGN_VARIABLE   │ 49,89      │ —
51    │ RESHAPE           │ 88,19      │ 90
52    │ FULLY_CONNECTED   │ 90,21,20   │ 91
53    │ LOGISTIC          │ 91         │ 92
54    │ QUANTIZE          │ 92         │ 93

  Op frequencies (Subgraph 0):
Op                │ Count
──────────────────┼──────
ASSIGN_VARIABLE   │ 6
CALL_ONCE         │ 1
CONCATENATION     │ 8
CONV_2D           │ 5
DEPTHWISE_CONV_2D │ 6
FULLY_CONNECTED   │ 1
LOGISTIC          │ 1
QUANTIZE          │ 1
READ_VARIABLE     │ 6
RESHAPE           │ 2
SPLIT_V           │ 2
STRIDED_SLICE     │ 10
VAR_HANDLE        │ 6

  ── Subgraph 1 ──────────────────────────────────
  Note: Subgraph 1 operations from flatbuffer parser (tflite package).
Index │ Op              │ Inputs │ Outputs
──────┼─────────────────┼────────┼────────
0     │ VAR_HANDLE      │ —      │ 0
1     │ ASSIGN_VARIABLE │ 0,1    │ —
2     │ VAR_HANDLE      │ —      │ 2
3     │ ASSIGN_VARIABLE │ 2,3    │ —
4     │ VAR_HANDLE      │ —      │ 4
5     │ ASSIGN_VARIABLE │ 4,5    │ —
6     │ VAR_HANDLE      │ —      │ 6
7     │ ASSIGN_VARIABLE │ 6,7    │ —
8     │ VAR_HANDLE      │ —      │ 8
9     │ ASSIGN_VARIABLE │ 8,9    │ —
10    │ VAR_HANDLE      │ —      │ 10
11    │ ASSIGN_VARIABLE │ 10,11  │ —

  Op frequencies (Subgraph 1):
Op              │ Count
────────────────┼──────
ASSIGN_VARIABLE │ 6
VAR_HANDLE      │ 6

Variable Tensors
────────────────
Subgraph │ Index │ Name                                 │ Shape          │ Dtype
─────────┼───────┼──────────────────────────────────────┼────────────────┼───────
0        │ 44    │ stream/states                        │ []             │ object
0        │ 45    │ stream_1/states                      │ []             │ object
0        │ 46    │ stream_2/states                      │ []             │ object
0        │ 47    │ stream_3/states                      │ []             │ object
0        │ 48    │ stream_4/states                      │ []             │ object
0        │ 49    │ stream_5/states                      │ []             │ object
0        │ 51    │ model/stream_1/concat/ReadVariableOp │ [1, 4, 1, 32]  │ int8
0        │ 52    │ model/stream_2/concat/ReadVariableOp │ [1, 10, 1, 64] │ int8
0        │ 53    │ model/stream_3/concat/ReadVariableOp │ [1, 14, 1, 64] │ int8
0        │ 54    │ model/stream_4/concat/ReadVariableOp │ [1, 22, 1, 64] │ int8
0        │ 55    │ model/stream_5/concat/ReadVariableOp │ [1, 5, 1, 64]  │ int8
0        │ 56    │ model/stream/concat/ReadVariableOp   │ [1, 2, 1, 40]  │ int8
0        │ 58    │ model/stream/strided_slice2          │ [1, 2, 1, 40]  │ int8
0        │ 61    │ model/stream_1/strided_slice2        │ [1, 4, 1, 32]  │ int8
0        │ 65    │ model/stream_2/strided_slice1        │ [1, 10, 1, 64] │ int8
0        │ 75    │ model/stream_3/strided_slice1        │ [1, 14, 1, 64] │ int8
0        │ 85    │ model/stream_4/strided_slice1        │ [1, 22, 1, 64] │ int8
0        │ 89    │ model/stream_5/strided_slice2        │ [1, 5, 1, 64]  │ int8
1        │ 2     │ stream_4/states1                     │ []             │ object
1        │ 4     │ stream_3/states1                     │ []             │ object
1        │ 6     │ stream_2/states1                     │ []             │ object
1        │ 8     │ stream_1/states1                     │ []             │ object
1        │ 10    │ stream/states1                       │ []             │ object

Quantization Summary
────────────────────
Quantized tensors:      75
Non-quantized tensors:  32
Dtype  │ Tensor Count
───────┼─────────────
int32  │ 31
int8   │ 63
object │ 12
uint8  │ 1

Memory Summary
──────────────
Subgraph │ Theoretical Tensor Memory (bytes)
─────────┼──────────────────────────────────
0        │ 41771
1        │ 3520
Total theoretical memory: 45291 bytes
This is theoretical minimum. Actual runtime memory depends on buffer reuse strategy.

Inference Test
──────────────
Status:      pass
Message:     Interpreter invocation succeeded and outputs were retrieved.
Input shape: [1, 3, 40]
Input dtype: int8
Index │ Name                      │ Shape  │ Dtype │ Min │ Max │ Mean
──────┼───────────────────────────┼────────┼───────┼─────┼─────┼─────
93    │ StatefulPartitionedCall:0 │ [1, 1] │ uint8 │ 5.0 │ 5.0 │ 5.0