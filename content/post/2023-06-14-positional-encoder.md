---
title: Position Encoding
date: 2023-06-14
tags: ["Transformer", "Positional Encoding", "PE"]
---

- [Giới Thiệu](#giới-thiệu)
- [Positional Encoding](#positional-encoding)
  - [Tại Sao Cần Có Positional Encoding ?](#tại-sao-cần-có-positional-encoding-)
  - [Khái niệm](#khái-niệm)
- [Cơ Chế Hoạt Động Của Positional Encoding](#cơ-chế-hoạt-động-của-positional-encoding)
- [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

<style>
  .img {
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  .imgTitle {
    text-align: center;
  }
</style>

# Giới Thiệu

Trong kiến trúc Transformer, trước khi Vector Embedding được đưa vào mô hình Encoder, nó được cộng thêm một vector khác để lưu trữ lại vị trí của các từ trong câu, cơ chế này gọi là Positional Encoding (PE).

<img class="img" src="/img/Transformer/PositionalEncoding/pe.jpg"/>

Để lưu trữ lại vị trí của các từ trong câu, thường có 2 hướng có thể tiếp cận là:

* Cho model học như học một vector embedding bình thường và lưu trữ các vị trí của các từ đó.
* Sử dụng một hàm số định nghĩa từ trước để lưu trữ và ánh xạ lại vị trí của các từ trong câu.

Trong bài báo gốc về Transformer, cơ chế PE được nhóm tác giả sử dụng là sử dụng một hàm được định nghĩa từ trước để ánh xạ và lưu trữ vị trí của các từ trong câu.

Và trong bài viết này chúng ta sẽ đi vào tìm hiểu cơ chế lưu trữ và cách thức hoạt động của hàm số đó trong kiến trúc Transformer.

# Positional Encoding

## Tại Sao Cần Có Positional Encoding ?

Không giống như các mạng hồi quy như RNN, LSTM, GRU hay phép tích chập Convolution có đươc thông tin của các context xung quanh.

Transformer chỉ sử dụng self-attention, do vậy nếu không có thông tin về vị trí thì output của kiến trúc transformer sẽ dễ bị đảo lộn vị trí giữa các token với nhau.

## Khái niệm

Positional Encoding là một cơ chế giúp mô hình có thể biết được ký tự, từ đang xét nằm ở vị trí nào trong câu.

Hàm Sinusoid là hàm được nhóm tác giả trong paper gốc sử dụng để lưu trữ vị trí.

Cụ thể tại mỗi ký tự hoặc từ sẽ được biểu diễn bởi 1 vector (position vector) với **\\(  d  \\)** chiều biểu diễn cho vị trí của ký tự, từ đó trong câu.

Để tính được **position vector** nhóm tác giả đưa ra phương pháp tính theo dạng:

\\[ PE_{(pos, d_{j})} = \sin(\frac{pos}{n^{\frac{2i}{d}}}) ~~ , ~~ subject~to:~~ d_{j}~is~even ~~ \\& ~~ d_{j} = 2i \\]

\\[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(1)\\]

\\[ PE_{(pos, d_{j})} = \cos(\frac{pos}{n^{\frac{2i}{d}}}) ~~ , ~~ subject~to:~~ d_{j}~is~odd ~~ \\& ~~ d_{j} = 2i + 1 \\]

Tức, để tính được position vector với d chiều, ta thực hiện tính giá trị tại từng chiều \\( d_{j} \\). Trong đó, tại vị trí **\\( i \\) chẵn** ta sẽ dùng hàm **sin** và tại vị trí **\\( i \\) lẻ** ta sẽ dùng hàm **cosin** và **n** là hyper-param (trong bài báo gốc nhóm tác giả chọn n=10000).

# Cơ Chế Hoạt Động Của Positional Encoding

Giả sử, ta có câu đầu vào là **"tôi đi học"** và số chiều cho position vector là \\( d = 15 \\). Thì khi đó ứng với vị trí của từng token và từng chiều trong số 15 chiều của position vector

Ta có:

<img class="img" src="/img/Transformer/PositionalEncoding/position_matrix.png"/>
<p class="imgTitle" >Bảng 1: Giá trị của từng token trên từng chiều của position vector</p>

Thế thì, cái bảng ở trên có ý nghĩa gì, và nó liên quan gì tới việc lưu trữ thông tin vị trí trong câu.

Để có cái nhìn tổng quát hơn, thì nếu ta coi vị trí trong câu (**pos**) là \\( x~~(x \in [0, length\\_sentence]) \\), ta thực hiện vẽ các đồ thị hình **sin** và **cos** theo công thức (1) tương ứng với từng chiều trong position vector.

Khi đó:

<img class="img" src="/img/Transformer/PositionalEncoding/visualize_detail/viewdetail_1.png"/>
<img class="img" src="/img/Transformer/PositionalEncoding/visualize_detail/viewdetail_2.png"/>
<p class="imgTitle">Hình 1: Tọa độ của các token trên đồ thị sinucoid theo các chiều trong position vector</p>


Từ công thức (1) cũng như Hình 1. Có thể thấy, trong một position vector \\( d \\) chiều, **\\( i \\)** sẽ tăng dần cho tới khi  \\( d_{j} = d \\) (tức  \\( d_{j} \\) là chiều cuối cùng của position vector như trong ví dụ trên thì  \\( d_{j} = 15 \\))

Thì khi đó biểu thức  \\( \frac{pos}{10000^{\frac{2i}{d}}} \\) sẽ **giảm dần** mỗi khi **i tăng**, điều này cũng đồng nghĩa với việc chu kỳ lượng giác của mỗi đồ thị **sin** và **cos** tương ứng trong (1) sẽ ngày càng **lớn hơn**.

Như Hình 1, \\( d_{j} = 4,5 \\) có chu kỳ lớn hơn  \\( d_{j} = 2,3 \\), và  \\( d_{j} = 2,3 \\) có chu kỳ lớn hơn  \\( d_{j} = 0,1 \\) ...v.v. (Lưu ý:  \\( d_{j} = 0,1 \\) có chu kỳ bằng nhau vì đều có  \\( i = 0 \\) tương tự các cặp khác)

Khi \\( d_{j} = 12,13 \\) đồ thị sẽ có dạng:

<img class="img" src="/img/Transformer/PositionalEncoding/visualize_detail/viewdetail_4.png">
<p class="imgTitle">Hình 2: Tọa độ của các token trên đồ thị sinucoid ở các chiều gần cuối trong position vector</p>

Khi \\( d_{j} \\) càng tiệm cận d, thì chu kỳ lượng giác lớn tới mức dường như các token ở các vị trí nhỏ sẽ gần như có giá trị bằng 0 đối với hàm sin và bằng 1 đối với hàm cos.

Một token phải ở vị trí đủ lớn (tức là một từ nằm ở vị trí nào đó cực xa trong 1 câu cực dài, tuy nhiên điều này cũng sẽ phụ thuộc phần lớn nữa là vào số chiều của position vector có lớn hay không (tức d có lớn hay không)).

Những nhận định trên là những nhận định đóng góp cực kỳ quan trọng của việc biểu diễn vị trí của token trong câu thông qua hàm **sinocoid** này của nhóm tác giả.

Từ những nhận định trên, ta có thể thấy, nếu 1 token càng ở **gần đầu câu** thì trong một position vector sẽ càng có **ít** sự biến thiên của dữ liệu, mà thay vào đó số lượng giá trị **0 và 1** sẽ **lặp đi lặp lại nhiều** lần.

Còn token càng ở **phần đuôi** của câu, thì số lượng giá trị **0 và 1** sẽ **ít hơn** mà thay vào đó sẽ là các giá trị khác (Điều này dễ thấy vì token càng xa thì sẽ càng gần đỉnh của đồ thị hơn là so với token gần khi mà đồ thị có chu kỳ lượng giác lớn).

Đây cũng là ý tưởng chính và cách thức lưu trữ vị trí của token trong câu của hàm sinocoid.

Ví dụ, như Hình 3 bên dưới, có \\( d = 15 \\), thì tại  \\( d_{j} = 12, 13 \\), dường như các token nằm ở gần đầu câu như  \\( t_{1},\ t_{4},\ t_{7} \\) lúc này chỉ nhận được 0 và 1 và không nhận được giá trị nào khác nữa.

Trong khi đó, các token ở xa hơn như  \\( t_{18},\ t_{k} \\) và xa nhất là  \\( t_{n} \\) các giá trị vẫn đang biến thiên thay vì nhận giá trị 0 và 1.

<img class="img" src="/img/Transformer/PositionalEncoding/visualize_detail/viewdetail_5.png"/>
<p class="imgTitle">Hình 3: Visualize vị trí của các token trong một câu dài trên đồ thị sinucoid ở các chiều gần cuối</p>

Có một số cách visualize để có cái nhìn tổng quan về sự biến thiên giá trị trong position vector, để dễ hiểu hơn cách thức lưu trữ giá trị của hàm sinocoid, dưới đây là một ví dụ:

<img class="img" src="/img/Transformer/PositionalEncoding/overview_PE.png"/>
<p class="imgTitle">Hình 4: <a  href="https://kazemnejad.com/blog/transformer_architecture_positional_encoding/">Visualize position encoding d=128 với max_length_sentence=50</a></p>

# Tài Liệu Tham Khảo

[1] <a href="https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers">Understanding Positional Encoding in Transformers - Kemal Erdem<a/>

[2] <a href="https://arxiv.org/abs/1706.03762">Ashish Vaswani et al, “Attention Is All You Need”, NeurIPS 2017</a>
