# 1. Channels and Kernels:
Channel is the depth of the input image. <br>
Kernel is a nxn dimension matrix with the help of which we do convolution process. Depth of the kernel depends on the input image depth. </n>
In general we use 3x3 kernel. This kerner reduces the size of output by 2 if size of stride is 1.

In the convolution process we convolve input image with the given kernel andf the depth or channel of the output will be equal to no of kernels used.

# 2. Why should we (nearly) always use 3x3 kernels?
This is the most fundamental kernel we can have. 3x3 kernels has the line of symmetry. 
We can create 2x2 kernel with the help of 3x3, just by putting "0" in the outer shell. We can same same receptive field 
with 3x3 as compare to higher dimension channel with very less no of parameters. 

# 3. How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)

We need to convolve 99 times to reach at 1x1.

199 x 199 | 3x3 > 197 x 197 <br>
197 x 197 | 3x3 > 195 x 195 <br>
195 x 195 | 3x3 > 193 x 193 <br>
193 x 193 | 3x3 > 191 x 191 <br>
191 x 191 | 3x3 > 189 x 189 <br>
189 x 189 | 3x3 > 187 x 187 <br>
187 x 187 | 3x3 > 185 x 185 <br>
185 x 185 | 3x3 > 183 x 183 <br>
183 x 183 | 3x3 > 181 x 181 <br>
181 x 181 | 3x3 > 179 x 179 <br>
179 x 179 | 3x3 > 177 x 177 <br>
177 x 177 | 3x3 > 175 x 175 <br>
175 x 175 | 3x3 > 173 x 173 <br>
173 x 173 | 3x3 > 171 x 171 <br>
171 x 171 | 3x3 > 169 x 169 <br>
169 x 169 | 3x3 > 167 x 167 <br>
167 x 167 | 3x3 > 165 x 165 <br>
165 x 165 | 3x3 > 163 x 163 <br>
163 x 163 | 3x3 > 161 x 161 <br>
161 x 161 | 3x3 > 159 x 159 <br>
159 x 159 | 3x3 > 157 x 157 <br>
157 x 157 | 3x3 > 155 x 155 <br>
155 x 155 | 3x3 > 153 x 153 <br>
153 x 153 | 3x3 > 151 x 151 <br>
151 x 151 | 3x3 > 149 x 149 <br>
149 x 149 | 3x3 > 147 x 147 <br>
147 x 147 | 3x3 > 145 x 145 <br>
145 x 145 | 3x3 > 143 x 143 <br>
143 x 143 | 3x3 > 141 x 141 <br>
141 x 141 | 3x3 > 139 x 139 <br>
139 x 139 | 3x3 > 137 x 137 <br>
137 x 137 | 3x3 > 135 x 135 <br>
135 x 135 | 3x3 > 133 x 133 <br>
133 x 133 | 3x3 > 131 x 131 <br>
131 x 131 | 3x3 > 129 x 129 <br>
129 x 129 | 3x3 > 127 x 127 <br>
127 x 127 | 3x3 > 125 x 125 <br>
125 x 125 | 3x3 > 123 x 123 <br>
123 x 123 | 3x3 > 121 x 121 <br>
121 x 121 | 3x3 > 119 x 119 <br>
119 x 119 | 3x3 > 117 x 117 <br>
117 x 117 | 3x3 > 115 x 115 <br>
115 x 115 | 3x3 > 113 x 113 <br>
113 x 113 | 3x3 > 111 x 111 <br>
111 x 111 | 3x3 > 109 x 109 <br>
109 x 109 | 3x3 > 107 x 107 <br>
107 x 107 | 3x3 > 105 x 105 <br>
105 x 105 | 3x3 > 103 x 103 <br>
103 x 103 | 3x3 > 101 x 101 <br>
101 x 101 | 3x3 > 99 x 99 <br>
99 x 99 | 3x3 > 97 x 97 <br>
97 x 97 | 3x3 > 95 x 95 <br>
95 x 95 | 3x3 > 93 x 93 <br>
93 x 93 | 3x3 > 91 x 91 <br>
91 x 91 | 3x3 > 89 x 89 <br>
89 x 89 | 3x3 > 87 x 87 <br>
87 x 87 | 3x3 > 85 x 85 <br>
85 x 85 | 3x3 > 83 x 83 <br>
83 x 83 | 3x3 > 81 x 81 <br>
81 x 81 | 3x3 > 79 x 79 <br>
79 x 79 | 3x3 > 77 x 77 <br>
77 x 77 | 3x3 > 75 x 75 <br>
75 x 75 | 3x3 > 73 x 73 <br>
73 x 73 | 3x3 > 71 x 71 <br>
71 x 71 | 3x3 > 69 x 69 <br>
69 x 69 | 3x3 > 67 x 67 <br>
67 x 67 | 3x3 > 65 x 65 <br>
65 x 65 | 3x3 > 63 x 63 <br>
63 x 63 | 3x3 > 61 x 61 <br>
61 x 61 | 3x3 > 59 x 59 <br>
59 x 59 | 3x3 > 57 x 57 <br>
57 x 57 | 3x3 > 55 x 55 <br>
55 x 55 | 3x3 > 53 x 53 <br>
53 x 53 | 3x3 > 51 x 51 <br>
51 x 51 | 3x3 > 49 x 49 <br>
49 x 49 | 3x3 > 47 x 47 <br>
47 x 47 | 3x3 > 45 x 45 <br>
45 x 45 | 3x3 > 43 x 43 <br>
43 x 43 | 3x3 > 41 x 41 <br>
41 x 41 | 3x3 > 39 x 39 <br>
39 x 39 | 3x3 > 37 x 37 <br>
37 x 37 | 3x3 > 35 x 35 <br>
35 x 35 | 3x3 > 33 x 33 <br>
33 x 33 | 3x3 > 31 x 31 <br>
31 x 31 | 3x3 > 29 x 29 <br>
29 x 29 | 3x3 > 27 x 27 <br>
27 x 27 | 3x3 > 25 x 25 <br>
25 x 25 | 3x3 > 23 x 23 <br>
23 x 23 | 3x3 > 21 x 21 <br>
21 x 21 | 3x3 > 19 x 19 <br>
19 x 19 | 3x3 > 17 x 17 <br>
17 x 17 | 3x3 > 15 x 15 <br>
15 x 15 | 3x3 > 13 x 13 <br>
13 x 13 | 3x3 > 11 x 11 <br>
11 x 11 | 3x3 > 9 x 9 <br>
9 x 9 | 3x3 > 7 x 7 <br>
7 x 7 | 3x3 > 5 x 5 <br>
5 x 5 | 3x3 > 3 x 3 <br>
3 x 3 | 3x3 > 1 x 1 <br>



# 4. How are kernels initialized? 
Kernels are initialized randomly. But we can initialized using certain rule like (from uniform or gaussian distribution).

# 5. What happens during the training of a DNN?
When the training start all the parameter started with some random value. DNN find the output and compare with the actual output.
It calculate loss. And DNN start modifying the value of all the parameters and again evaluate the output with the new value of
parameters and again find the loss and keep updating the parameters until DNN closed to the actual output.
