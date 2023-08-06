---
title: Self-Attention
date: 2023-06-15
tags: ["Transformer", "Self-Attention", "Attention"]
---

- [Giới Thiệu](#giới-thiệu)
- [Self-Attention](#self-attention)
  - [Ý Tưởng](#ý-tưởng)
  - [Cơ Chế Hoạt Động](#cơ-chế-hoạt-động)
    - [Self-Attention](#self-attention-1)
      - [Ma Trận Query, Key, Value (Q, K, V Matrix)](#ma-trận-query-key-value-q-k-v-matrix)
      - [Tên Gọi Của Các Attention Khi Được Áp Dụng Tại Các Vị Trí Kháu Trong Transformer](#tên-gọi-của-các-attention-khi-được-áp-dụng-tại-các-vị-trí-kháu-trong-transformer)
      - [Multi-Head Attention](#multi-head-attention)
  - [Attention Và Self-Attention](#attention-và-self-attention)
- [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

<style>
  .img {
    width: 500px;
    height: 500px;
    display: block;
    margin-left: auto;
    margin-right: auto;

  }
  .scaleImg {
    width: 700px;
    height: 500px;
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  .imgTitle {
    text-align: center;
  }
</style>

# Giới Thiệu

Ta đã đi qua tổng quan về ý tưởng của transformer, ta cũng đã bắt đầu đi sâu vào trong kiến trúc để hiểu cơ chế hoạt động với việc tìm hiểu Positional Encoding.

Trong bài viết này ta sẽ đi tiếp tới cơ chế tiếp theo, có thể nói là cơ chế quan trọng nhất trong kiến trúc Transformer là Self-Attention.

Nói không ngoa thì Self-Attention là linh hồn của kiến trúc transformer. Mọi thứ mới mẻ trong kiến trúc transformer được công bố tại thời điểm bấy giờ được sinh ra và xoay quanh Self-Attention.

<img class="img" src="/img/Transformer/SelfAttention/sa_pos_in_transformer.png"/>
<p class="imgTitle">Hình 1: Vị trí của self-attention trong kiến trúc transformer</p>

# Self-Attention
## Ý Tưởng

Ví dụ, khi ta có một câu đầu vào với n token \\( t_{1},\ t_{2},\ t_{3},...,\ t_{n} \\) . Khi đó, một cách dễ hiểu self-attention là cơ chế giúp token \\( t_{i} \\)  chú ý tới các token còn lại trong câu, để có thể giúp token \\( t_{i} \\)  nắm được mối quan hệ của nó với các token còn lại về mặt cấu trúc câu, mặt ngữ nghĩa, ..v.v. là như thế nào.

## Cơ Chế Hoạt Động

Trong bài báo gốc, self-attention được nhóm tác giả giới thiệu với một loạt các khái niệm khác liên quan như **Scaled dot-product attention** hay **Multi-head attention**.

Trong đó **Scaled dot-product attention** là một cơ chế self-attention và **Multi-head attention** là việc nối nhiều **scaled dot-product attention** lại với nhau và đưa qua một lớp **fully connected**.

<img class="scaleImg" src="/img/Transformer/SelfAttention/sa_architecture.png"/>
<p class="imgTitle">Hình 2: Self-Attention trong transformer</p>

### Self-Attention

#### Ma Trận Query, Key, Value (Q, K, V Matrix)

Trên Hình 2, có thể thấy Q, K, V cũng là 3 tham số được giới thiệu trong self-attention. Vậy Q, K, V là gì và đóng vai trò như thế nào trong self-attention.

Q, K, V là 3 vector đại diện biểu diễn cho từng token trong câu được tạo ra bằng cách nhân ma trận biểu diễn các token đầu vào với 3 ma trận học tương ứng là \\( W^{Q}\ W^{K}\ W^{V} \\) .

<img class="scaleImg" src="/img/Transformer/SelfAttention/sa_qkv.jpg">
<p class="imgTitle">Hình 3: Ma trận Q, K, V trong self-attention</p>

Trong đó:

* **Q**: Query vector dùng để chứa thông tin của câu tìm kiếm (ví dụ như chứa các thông tin của token đang cần xem xét).
* **K**: Key vector dùng để biểu diễn thông tin so sánh giữa các token trong câu với token đang được query.
* **V**: value vector biểu diễn nội dung của các token.

Để dễ hiểu hơn, nếu câu đầu vào là **"tôi đi học"** với số chiều embedding là \\( d=100 \\)  thì **Q, K, V** sẽ được biểu diễn dưới dạng ma trận có \\( (3 \times 100) \\) .

Khi đó, ma trận \\( attention \\_ score = softmax(\frac{QK^{T}}{\sqrt[2]d_{k}}) \\)  có thể được biểu diễn dưới dạng visulaize:

<img class="img" src="/img/Transformer/SelfAttention/attention_score.png">
<p class="imgTitle">Hình 4: Visualize cách thức ma trận attention_score biểu diễn mối quan hệ giữa các token trong câu</p>

Có thể thấy việc thực hiện \\( attention \\_ score = softmax(\frac{QK^{T}}{\sqrt[2]d_{k}}) \\)  sẽ giúp cho mô hình có thể học được mối quan hệ của các từ trong câu, như ("tôi-tôi", "tôi-đi", "tôi-học", "đi-tôi", "đi-đi", "đi-học" ..v.v..).

Tuy nhiên, với đó thì chưa đủ vì bản chất đó chỉ là học các mối liên hệ giữa các token trong câu nhưng không giữ được giá trị, ý nghĩa của cả câu ban đầu là **"tôi đi học"** mang giá trị gì.

Thì khi đó **V** ở đây để giữ nguyên giá trị, ý nghĩa của câu đầu vào đó để kết hợp với attention_score và tạo thành một biểu thức self-attention hoàn chỉnh, thứ mà giúp model vừa có thể hiểu giá trị, ý nghĩa tổng quan của cả câu đầu vào vừa có thể hiểu mối quan hệ giữa các token trong câu với nhau.

Khi đó, ta có công thức cho attention là:

\\[ Self \- Attention \ Output(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt[2]d_{k}})V \\]

Đây được coi là hàm số để tính điểm cho attention, tham số \\( \sqrt[2]d_{k} \\)  xuất hiện ở đây với mục đích scale nhỏ lại bộ giá trị ở tử số trong hàm softmax.

Nếu giá trị \\( QK^{T} \\)  là một vector lớn và không chia cho tham số \\( \sqrt[2]d_{k} \\) , thì khi đó với tính chất của hàm mũ trong hàm softmax là \\( \frac{e^{z_{i}}}{\sum_{j}^{nclass}e^{z_{j}}} \\)  thì input càng lớn sẽ càng khiến giá trị lớn nhất trong input tiến tới 1 và các giá trị còn lại tiến dần tới 0.

Ví dụ, nếu như ta có một hàm softmax cho 5 class với input bất kỳ, ta thực hiện scale từ nhỏ đến lớn thì khi đó đồ thị phân bố xác suất cho hàm softmax đó có dạng:

<img class="img" src="/img/Transformer/SelfAttention/scale_input_softmax.png">
<p class="imgTitle">Hình 5: Đồ thị phân bố xác suất của hàm softmax khi scale input</p>

Thì khi đó các class có giá trị sau khi đi qua hàm softmax tiến tới 0 khi train trong quá trình backpropagation sẽ xảy ra hiện tượng **vanishing gradient** và sẽ không đóng góp gì nhiều giá trị học trong quá trình train.

Do vậy, nhóm tác giả thực hiện scale nhỏ lại input của hàm softmax bằng tham số \\( \sqrt[2]d_{k} \\) để giúp cho các class khác mặc dù vẫn sẽ thấp nhưng không bị thấp quá.

#### Encoder, Decoder, Cross Attention

<img class="img" src="/img/Transformer/SelfAttention/sa_type.png">
<p class="imgTitle">Hình 6: Tên gọi của các attention khi được áp dụng tại các vị trí khác nhau</p>

Về mặt bản chất các Attention trong Transformer có chung một cơ chế là Self-Attention như ở phần trước.

Tuy Nhiên, khác với encoder và decoder attention nhận đầu vào là các represent vector của câu đầu vào được đi qua embedding layer và cộng với position vector, thì cross attention layer nhận đầu vào từ encoder và decoder để học mối quan hệ giữa 2 phần.


#### Multi-Head Attention

<img class="scaleImg" src="/img/Transformer/SelfAttention/sa_qkv_multi.jpg">
<p class="imgTitle">Hình 7: Ma trận Q, K, V trong multi-head attention</p>


Về cơ bản Multi-head attention có thể được định nghĩa là việc sử dụng nhiều lớp self-attention rồi nối chúng lại với nhau, sau đó nhân với một ma trận học \\( W^{O} \\)

Thông thường, để hiểu được vai trò của một từ trong câu, nó sẽ cần được nhìn ở nhiều khía cạnh khác nhau như cấu trúc câu, ngữ nghĩa, ..v.v.

Thì thay vì chỉ sử dụng 1 self-attention hay còn gọi là 1 head thì nhóm tác giả sử dụng nhiều self-attention hay multi-head để mỗi head sẽ tập trung vào học một khía cạnh khác nhau.

**Note**: Thứ mà model học là các ma trận \\( W^{Q}\,\ W^{K}\,\ W^{V}\,\ W^{O}\\) Chứ **không phải** các ma trận **Q, K, V, Z**. Đừng nhầm lẫn nhé

## Attention Và Self-Attention

Trước đó, với mô hình seq2seq, ở đâu đó ta đã biết qua một số cơ chế **attention** khác nhau để giúp seq2seq model có sự chú ý tới các token trong câu có thể đến như **Bahdanau** và **Luong Attention**.

Sau đây ta sẽ đi qua một nhận định giữa Attention và Self-Attention theo góc nhìn và hiểu biết của mình.

* Với các neural network ta sẽ đưa input (câu đầu vào) qua các layer và các activate fuction, và trong các mạng RNN và biến thể của nó thì ta sẽ có thêm state của các layer nữa. Thì khi đó Attention được áp dụng sẽ nhận đầu vào là các input đã đi qua các layer và các activate fuction. Trong khi đó Self-Attention sẽ thực hiện attention tại chính câu đầu vào ở mỗi layer có sử dụng nó.

* Attention thường được áp dụng để giúp bộ phận decoder có thể có thêm thông tin về phía encoder. Self-Attention có thể hoạt động độc lập trên cả 2 bộ phân encoder và decoder mà không có sự kết nối nào ở đây. Do hoạt động độc lập nên có cả các biến thể của Transformer được sinh ra khi chỉ dùng Encoder hoặc Decoder (BERT là một ví dụ điển hình).

# Tài Liệu Tham Khảo

[1] <a href="https://arxiv.org/abs/1706.03762">Ashish Vaswani et al, “Attention Is All You Need”, NeurIPS 2017</a>

[2] <a href="https://theaisummer.com/self-attention/">Why multi-head self attention works: math, intuitions and 10+1 hidden insights - Nikolas Adaloglou</a>

[3] <a href="https://jalammar.github.io/illustrated-transformer/">The Illustrated Transformer - Jay Alammar</a>

[4] <a href="https://towardsdatascience.com/transformer-networks-a-mathematical-explanation-why-scaling-the-dot-products-leads-to-more-stable-414f87391500">Transformer Networks: A mathematical explanation why scaling the dot products leads to more stable gradients</a>
