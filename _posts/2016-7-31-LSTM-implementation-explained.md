---
layout: post
title: LSTM实现解析
---

## RNN
RNN（循环神经网络）和前馈神经网络区别在于RNN的隐含层单元有层内的连接，数学上体现在：隐层单元的输出既取决于前一层的输入也由同一层的隐层单元决定。
![](/images/lstm/fnn&rnn.png)
\\[a_t=Ux_t+Ws_{t-1}+b\\]
\\[s_t=tanh(a_t)\\]
\\[o_t=Vs_t+c\\]

## LSTM
LSTM设计用于解决长期依赖问题。一个LSTM单元设计由3个门（`input gate`, `foget gate`, `output gate`）以及1个细胞单元（cell unit）组成。

Gates:
\\[i_{t} = g(W_{xi}x_{t} + W_{hi}h_{t-1} + b_{i})\\]
\\[f_{t} = g(W_{xf}x_{t} + W_{hf}h_{t-1} + b_{f})\\]
\\[o_{t} = g(W_{xo}x_{t} + W_{ho}h_{t-1} + b_{o})\\]
Input transform:
\\[c\_in_{t} = tanh(W_{xc}x_{t} + W_{hc}h_{t-1} + b_{c\_in})\\]
State update:
\\[c_{t} = f_{t} \cdot c_{t-1} + i_{t} \cdot c\_in_{t}\\]
\\[h_{t} = o_{t} \cdot tanh(c_{t})\\]

三个门的激活函数一般都是sigmoid函数，输出的介于[0,1]之间的值决定保留或者遗忘当前Cell state的多少。Cell state的更新由\\(f_t\\)决定遗忘多少先前的状态，由\\(i_t\\)决定接受多少当前输入。
![](/images/lstm/cell.svg)

LSTM能够解决长期依赖问题在于Cell state，在前面的公式表示中可以看到Cell state只有简单的「线性」变换，舍去旧值的一部分，以及加上输入的一部分。

## 实现LSTM层
代码由Torch7框架实现，并使用`nn`模块的以下层：
- nn.Identity - 构建输入空间
- nn.Dropout(p) - dropout模块
- nn.Linear(in, out) - 维度in到维度out的仿射变换
- nn.Narrow(dim, start, len) - 从start索引选取维数为dim的len个元素
- nn.Sigmoid() - Sigmoid函数
- nn.Tanh() - tanh函数
- nn.CMulTable() - 张量乘法
- nn.CAddTable() - 张量和

### 输入
LSTM层接受的输入张量为：
\\[\\{input, c_{t-1}^1, h_{t-1}^1\\}\\]

```lua
local inputs = {}
table.insert(inputs, nn.Identity()())   -- network input
table.insert(inputs, nn.Identity()())   -- c at time t-1
table.insert(inputs, nn.Identity()())   -- h at time t-1
local input = inputs[1]
local prev_c = inputs[2]
local prev_h = inputs[3]
```

### 计算门的值

```lua
local i2h = nn.Linear(input_size, 4 * rnn_size)(input)  -- input to hidden
local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)   -- hidden to hidden
local preactivations = nn.CAddTable()({i2h, h2h})       -- i2h + h2h
```

Linear变换分为input to hidden的从input_size到4倍的rnn_size以及hidden to hidden的，在上面的公式中可以看到，激活值来自于三部分：input即公式中的\\(x_t\\),hidden即公式中的\\(h_{t-1}\\),以及偏置。以input to hidden来说，变换后的向量维度为4倍的rnn_size，分为4部分：第一部分是input gate，第二部分是foget gate，第三部分是output gate，第四部分是cell input。hidden to hidden同样如此。

![](/images/lstm/preactivation_graph.svg)

接下来计算经过激活函数的值，三个门使用的sigmoid函数，而cell input使用的是tanh函数。

```lua
-- gates
local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(preactivations)
local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)

-- input
local in_chunk = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(preactivations)
local in_transform = nn.Tanh()(in_chunk)
```
![](/images/lstm/gates.svg)

最后，分别把每一个门的值提取出来。
```lua
local in_gate = nn.Narrow(2, 1, rnn_size)(all_gates)
local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(all_gates)
local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(all_gates)
```
### 计算cell state和hidden state
按照公式，先计算下一状态的cell state，更新值来自两部分，一是前一时刻的cell state，另一个是输入。

![](/images/lstm/state_calculation.svg)

```lua
-- previous cell state contribution
local c_forget = nn.CMulTable()({forget_gate, prev_c})
-- input contribution
local c_input = nn.CMulTable()({in_gate, in_transform})
-- next cell state
local next_c = nn.CAddTable()({
  c_forget,
  c_input
})
```

对新的cell state做tanh激活，再和foget gate做外积，得到hidden state。

```lua
local c_transform = nn.Tanh()(next_c)
local next_h = nn.CMulTable()({out_gate, c_transform})
```

## Reference

[](https://apaszke.github.io/lstm-explained.html)
