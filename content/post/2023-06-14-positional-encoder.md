---
title: Positional Encoding
date: 2023-06-14
tags: ["Transformer", "Positional Encoding", "PE", "RoPE", "Rotary Positional Encoding", "ALiBi", "Attention with Linear Bias"]
---

- [Giới Thiệu](#giới-thiệu)
- [Positional Encoding](#positional-encoding)
  - [Tại Sao Cần Có Positional Encoding ?](#tại-sao-cần-có-positional-encoding-)
  - [Khái niệm](#khái-niệm)
- [Cơ Chế Hoạt Động Của Positional Encoding](#cơ-chế-hoạt-động-của-positional-encoding)
- [Các Phương Pháp Khác](#các-phương-pháp-khác)
  - [Relative Positional Encoding](#relative-positional-encoding)
    - [Attention with Linear Bias (ALiBi)](#attention-with-linear-bias-alibi)
  - [Rotary Positional Encoding (RoPE)](#rotary-positional-encoding-rope)
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

# Các Phương Pháp Khác

Ở trên là một trong những phương pháp encode position được biết đến rộng rãi. Vì là cách tiếp cận đi liên với kiến trúc gốc của Transformers. Và phương pháp trên thường được biết đến là thuộc nhóm cách tiếp cận **Absolute Positional Encoding**

Ngoài ra còn có các cách tiếp cận khác là **Learnable Positional Encoding** và **Relative Positional Encoding**

## Relative Positional Encoding

### Attention with Linear Bias (ALiBi)

Năm 2022, các nhà nghiên cứu của Facebook kết hợp cùng trường đại học Washington và viện nghiên cứu Allen đã đề xuất một phương pháp **positional encoding** khác bằng việc thêm một hàm Bias (độ lệch, b(i, j)) khi thực hiện tính toán attention score. Tên gọi của phương pháp này là **Attention with Linear Bias** (ALiBi).

Sau khi tích vô hướng của các ma trận query và key để có được điểm attention, một hàm Bias được thêm vào. Hàm Bias b(i, j) được thêm vào sẽ là các thuật toán khác nhau có thể là tuyến tính (thường được sử dụng) hoặc phi tuyến tùy thuộc vào mục đích khác nhau. Nhưng các hàm b(i, j) được thêm vào thường với mục đích để tăng cường sự chú ý của mô hình và cho phép mô hình tập trung hơn vào các vị trí cụ thể trong chuỗi.

Ví dụ, nếu mục đích chỉ muốn có thêm thông tin của vị trí của các token trong câu mà việc token đứng đâu trong câu không quá quan trọng, thì b(i, j) có thể chỉ đơn giản là hàm tính khoảng cách **Euclidean**. Nếu như việc 2 token trong câu đứng gần nhau sẽ mang lại sự tác động khác với 2 token đứng xa nhau chẳng hạn thì hàm **Log** có thể sử dụng để tính khoảng cách giữa 2 token, ..v..v.

### Rotary Positional Encoding (RoPE)

Theo đó vào năm 2023, các nhà nghiên cứu thuộc Zhuiyi Technology Co. LTD. Shenzhen đã đề xuất một phương pháp mã hóa vị trí tương đối khác dựa trên việc xoay embedding đầu vào nhằm giữ được tính toàn vẹn của vector embedding đầu vào đó mà vẫn có được thông tin vị trí "**tương đối**" của các token trong câu (mang lại độ hiệu quả và linh hoạt cao hơn các phương pháp tiếp cận trước đó) với tên gọi là **Rotary Positional Encoding** (RoPE).

RoPE sẽ thực hiện xoay mà không cộng hay thêm biến số nào vào các vector query và key khi thực hiện tính attention score (khác với phương pháp trước đó  (đã được nêu ở trên) sử dụng hàm sinusoidal là positional encoding vector sẽ được cộng trực tiếp vào input embedding). Mỗi một token tương ứng với từng vị trí khác nhau trong câu sẽ có một góc xoay khác nhau và là duy nhất.

Giả sử, ta có một câu với 3 token. Và 3 token đó tương ứng với 3 kim giây, phút, giờ trong đồng hồ. 3 kim này có tần số và tốc độ quay là khác nhau, khi quay ta có thể ước lượng được khoảng cách tương đối của chúng với nhau về việc kim nào gần kim nào hơn.

Vậy giả sử, với nhiều token hơn, và mỗi token có tần số hay tốc độ nhanh hơn nhau 1 đơn vị. Ta sẽ có,

<img class="img" src="/img/Transformer/PositionalEncoding/rope.png"/> 

Khi nhìn ta cũng có thể hình dung được luôn vị trí tương đối của chúng đối với nhau, token nào gần hay xa token nào hơn.

Ví dụ, cặp token 1-5 có **góc lớn hơn** cặp 1-3 => token 5 **xa** token 1 hơn token 3, nhưng cặp token 1-5 lại có **góc nhỏ hơn** cặp 1-6 => token 5 **gần** token 1 hơn token 6.

#### NOTE
* Đã có các nghiên cứu liên quan về việc mã hóa vị trí tuyệt đối như phương pháp sử dụng sinusoidal trước đó vẫn gặp phải một số hạn chế và hoạt động kém hiệu quả. Ví dụ, Nếu dữ liệu thực tế có độ dài khác biệt "tương đổi or lớn" đối với dữ liệu huấn luyện, thì mã hóa vị trí tuyệt đối sẽ trở nên kém hiệu quả.
* Trong khi đó RoPE với vieecj mã hóa vị trí tương đối xử dụng góc xoay, mang lại độ linh hoạt và hiệu quả cao hơn kể cả đối với sự khác biệt về độ dài của câu thực tế và dữ liệu huấn luyện.

* Sinusoidal encoding mất nhiều chi phí tính toán hơn khi phải tính thêm positional encoding vector và cộng vào input embedding.
* Trong khi, RoPE thực hiện tính toán theo từng cặp q, k vector trong lúc tính attention score. Thêm vào đó, việc không **cộng** thêm bất vector hay biến số nào cho input embedding, RoPE vẫn có được thông tin vị trí cho từng token trong câu mà vẫn đảm bảo được tính toàn vẹn của dữ liệu đầu vào (hay giá trị cho input embedding).

* RoPE sẽ thực hiện xoay theo từng cặp q, k vector khi tính attentio score. Về lý thuyết, việc thực hiện xoay toàn bộ vector q và k trước khi tính toán attention score là vẫn đúng, cả 2 đều giữ được tính đồng nhất về phép xoay và góc xoay đối với từng token trong câu. Tuy nhiên, việc thực hiện xoay cả vector q và k sẽ cần lưu trữ và xử lý ma trận xoay điều này sẽ phức tạp hơn khi triển khai và tốc đọ tính toán cũng lâu hơn thay vì thực hiện phép xoay khi tính attention score.

# Tài Liệu Tham Khảo

[1] <a href="https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers">Understanding Positional Encoding in Transformers - Kemal Erdem<a/>

[2] <a href="https://arxiv.org/abs/1706.03762">Ashish Vaswani et al, “Attention Is All You Need”, NeurIPS 2017</a>

[3] <a href="https://arxiv.org/pdf/2108.12409">Ofir Press, Noah A. Smith1, Mike Lewis, “TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION”, 2022</a>

[4] <a href="https://arxiv.org/pdf/2104.09864">, “ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING”, 2023</a>