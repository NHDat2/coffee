---
title: Activation Function
date: 2023-09-10
tags: ["activation function", "hàm kích hoạt"]
---

<style>
.textSingleImg {
  text-align: center;
}
.textTwoImg {
    display: flex;
    flex-direction: row;
    justify-content: space-around;

}
.singleImg {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.twoImg {
    display: inline;
    width: 300px;
    height: 300px;
    margin-left: 30px;
}
</style>

# Giới Thiệu

Activation function, ta thường được biết đến là các hàm “phi tuyến (non-linear)” được dùng “phổ biến” trong các Neural Network models, với mục đích là bẻ cong sự “tuyến tính” của các hàm số, của models, giúp cho models có thể học được các pattern phi tuyến.

Như một mạng ANN thông thường, nếu như không có sự can thiệp của các “activation function” thì cho dù có bao nhiêu hidden layer đi nữa thì sau cùng, mạng ANN đó cũng chỉ là “tổ hợp tuyến tính” => sẽ đưa ra 1 hàm tuyến tính khác và sẽ không thể mô hình hoá được các pattern phi tuyến.

Trong bài viết này, ta sẽ đi qua từ khái niệm, các loại activation function và sự hữu dụng của chúng.

# Activation Function

Như đã nói ở trên, Activation function là các hàm phi tuyến, được đưa vào để giúp cho các hàm tuyến tính có được đầu ra phi tuyến hay được sử dụng nhiều nhất trong các neural network để giúp models có thể học được các pattern phức tạp khác nhau.

Đương nhiên không phải hàm phi tuyến nào cũng có thể đưa ra áp dụng vào neural network ta cũng coi đó là “activation function”. Mà “activation function” sẽ còn tuân theo một số tiêu chí nhất định nữa.

Theo những kiến thức mà mình có được thì, một “activation function” nên có các tiêu chí sau:

* **Là hàm phi tuyến**
* **Khả vi** (vì thường sẽ dùng gradient để thực hiện tính backpropagation, nên việc hàm số phải tính được đạo hàm là bắt buộc)
* **Ưu tiên chi phí tính toán nhỏ** (khả vi thôi là chưa đủ, vì lượng activation function thường được sử dụng rất nhiều trong các neural network, do vậy, ta sẽ cần phải tính qua activation function rất nhiều lần)

Mình nghĩ trên là các tiêu chí cơ bản mà 1 activation function nên có, và đương nhiên sẽ có các hàm khác nhau được đưa ra mỗi hàm sẽ có sự khác nhau để phù hợp với dữ liệu, bài toán cụ thể or cải tiến của các hàm cũ để cho ra hiệu suất tốt hơn. Ta sẽ bàn ở phần tiếp theo.

# Các Activation Function

Mình nhận thấy, có một số vấn đề mà các “activation function” dễ gặp phải. Cũng như là mục tiêu để các activation khác nhau cố gắng cải thiện là:

* **Vanishing Gradient**: là hiện tượng khi tính backpropagation, mà gradient của các lớp có giá trị rất nhỏ gần 0, thì khi tính gradient ở các layer đầu thì sẽ phải nhân rất nhiều giá trị nhỏ với nhau dẫn đến gradient -> 0 (nhiều giá trị trong khoảng (0, 1) nhân với nhau sẽ ra số -> 0) => các trọng số mới sẽ không được cập Nhật gì thêm so với các trọng số cũ => Model không học được gì.

* **Exploding Gradient**: Ngược lại là hiện tượng khi tính backpropagation, mà gradient của các lớp có giá trị rất lớn, thì khi tính gradient ở các layer đầu thì sẽ phải nhân rất nhiều giá trị lớn với nhau dẫn đến gradient -> vô cự => các trọng số mới sẽ được cập Nhật rất lớn, 1 cục to đùng => Model có thể nhảy lung tung mà không tìm được global minimum

* **NOT being Zero Centered**: là hiện tượng giá trị output của các activation function không đối xứng qua gốc tọa độ 0 => gradient sẽ dễ bị xê dịch sang một hướng trái or phải => khi đó, output của các node trong neural network cũng sẽ có hiện tượng tương tự => Dẫn đến việc huấn luyện model sẽ không ổn định

Sẽ còn các vấn đề khác mà mình chưa tìm ra ở đây nữa, các bạn có thể tìm hiểu thêm nha.


## Sigmoid

Sigmoid là một activation function, đưa giá trị đầu ra về khoảng [0, 1]

\\[ Sigmoid = {1 \over 1 + e^{-x}} \\]
\\[ Sigmoid\ ' = sigmoid * (1 - sigmoid) \\]

<img class="singleImg" src="/img/activation/1.jpg">

Có thể thấy, đầu ra của sigmoid đưa giá trị về khoảng [0, 1], mà khi tính toán ta thường tính xác suất cho output và xác suất của bất kỳ thứ gì cũng sẽ thuộc khoảng [0, 1] => sigmoid rất phù hợp trong khoản này.

Và thực tế, công thức sigmoid chính là công thức để convert từ số thực sang dạng xác suất. Ta sẽ đi sâu hơn ở phần sau **“The odds”**

Tuy nhiên, cũng có một số draw-back mà sigmoid gặp phải:
* Có thể nhìn vào đồ thị đạo hàm của sigmoid ở trên, ta có thể thấy dường như đạo hàm chỉ có ý nghĩa khi \\( x \in [-3, 3] \\), còn lại giá trị đạo hàm rất nhỏ và tiến tới 0 khi \\(x\ \in\ [3, +\infty\) \\) hoặc \\(x\ \in\ [-3, -\infty\) \\) => dễ dẫn đến **vanishing gradient**
* Vì output của sigmoid trong khoảng [0, 1] => **NOT being zero centered**

## Tanh

Khá giống với Sigmoid, thay vì đưa giá trị output trong khoảng [0, 1] thì TANH đưa output thuộc [-1, 1]

\\[ Tanh = {e^x - e^{-x} \over e^x + e^{-x}} \\]
\\[ Tanh' = 1 - Tanh^2 \\]

<img class="singleImg" src="/img/activation/2.jpg">

Có thể thấy, các đồ thị của TANH khá giống với sigmoid, output của TANH đưa ra trong khoảng [-1, 1] đối xứng nhau qua gốc tọa độ => giúp việc huấn luyện model ổn định hơn

Tuy nhiên, từ đồ thị đạo hàm của TANH, ta có thể thấy TANH vẫn dễ gặp hiện tượng vanishing gradient giống với hàm sigmoid

## ReLU

ReLU được biết đến là một trong những activation function được sử dụng rộng rãi bậc nhất trong các deep learning models.

ReLU đưa giá trị output về khoảng \\( [0, +\infty\) \\).

ReLU sẽ chuyển tất cả các giá trị nhỏ hơn 0 từ giá trị đầu vào thành 0, và sẽ giữ nguyên các giá trị > 0. Với điều này, ReLU chỉ kích hoạt một số neuron nhất định thay vì toàn bộ neuron tại một thời điểm trong quá trình học.

\\[ ReLU = \max(0, x) \\]
\\[ ReLU' = \begin{cases} 1 & nếu\ \ x \gt 0 \\\\ 0 & nếu\ \ x \leq 0 \\ \end{cases} \\]

<img class="singleImg" src="/img/activation/3.jpg">

Vì ReLU là hàm số rất đơn giản, cùng với việc chỉ thực hiện kích hoạt đối với một số neuron nhất định => Ưu điểm lớn nhất của ReLU là việc chi phí tính toán rất nhanh và nhanh hơn rất nhiều so với các hàm mũ như sigmoid hay tanh.

ReLU cũng đã **giải quyết** vấn đề vanishing gradient mà dễ gặp ở Sigmoid hay Tanh function.

Tuy nhiên, ReLU cũng gặp draw-back được gọi là **“Dying ReLU Problem”**.

**Dying ReLU Problem**, mô tả việc vì 1 phần bên trái của hàm ReLU sẽ khiến gradient bằng 0, khi đó, sẽ vô hiệu hoá 1 phần các neuron => Do vậy, trong quá trình backpropagation sẽ khiến cho một số weight và bias của một số neuron sẽ không được update => Được gọi là dead neuron (khi mà các neuron sẽ không bao giờ được activate)

Hơn nữa, vì hiện tượng này, khi mà các giá trị <0 sẽ bị biến thành 0 ngay lập tức sẽ phần nào đó ảnh hưởng tới quá trình học vì dễ bị miss các pattern mà các neuron đó nắm giữ

## Softmax

Có thể nói SOFTMAX là phiên bản tổng quát hoá hơn của Sigmoid. Giống với sigmoid output của softmax nằm trong khoảng [0, 1].

Softmax thường được sử dụng ở layer cuối cho các bài toán multi-class. Khi mà kết quả đầu ra là xác suất của các class tương ứng, với tổng xác suất sẽ = 1

\\[ Softmax = {e^{x_i} \over \sum_{j} e^{x_j}} \\]

Tuy nhiên, softmax vẫn gặp một số vấn đề giống với sigmoid như **“vanishing gradient”** và **“Not being zero centered”**.

# The odds

The odds, được biết đến là bắt nguồn từ các vụ “cá cược trong đua ngựa” và “the odds” được biết tới là “tỉ lệ cược”.

\\[ the\ odds = {Something\ Happen \over Something\ Not\ Happen} \\]

**NOTE: The odds # probability**

Ví dụ, trong một giải bóng đá, Team A đã chơi 8 trận bóng, gồm 5 thắng và 3 thua.

Khi đó:

<img class="singleImg" src="/img/activation/4.jpg">

Tuy nhiên, nếu ta lấy **(1) / (2)** hoặc **(2) / (1)** ta sẽ có:

<img class="singleImg" src="/img/activation/5.jpg">

Thực hiện \\( log \\) cả 2 vế, ta có:

\\[ log(odds) = log({P \over 1 - P}) \\]

<img class="singleImg" src="/img/activation/6.jpg">

<p class="textSingleImg"><b>Đồ thị hàm số log(odds)</b></p>

Ta có thể thấy, vì \\( P \in [0, 1] \Rightarrow log(odds) \in \(-\infty, +\infty\) \\)

Ta đặt, \\( log\(odds\) = X, \forall X \in \(-\infty, +\infty\) \\)

Ta có:

<img class="singleImg" src="/img/activation/7.jpg">

=> ta có với đầu vào \\( X \in \(-\infty, +\infty\)\\) ta có thể convert về thành xác suất \\( P \in [0, 1] \\) và đó là cách mà sigmoid hoạt động.

Gần giống vậy, thì softmax có thể được coi là mở rộng, tổng quát hoá hơn của sigmoid.

Đó cũng là lý do tại sao, các hàm sigmoid hay softmax thường được xử dụng ở layer cuối của models.

# Thảo Luận

Đương nhiên còn rất nhiều các activation function khác như “Leaky ReLU, Swish, GELU, …v.v” trong bài viết này, mình chỉ đề cập 1 số hàm thôi và m.n có thể tìm hiểu thêm về các hàm khác nữa.

Ta cũng có thể thấy, thường các hàm dễ gặp phải các hiện tượng “vanishing or exploding gradient” sẽ được tránh or ít khi được sử dụng cho các “hidden layer”, ví dụ như “Sigmoid, Tanh, Softmax, ELU, ..v.v” và các hàm phức tạp như hàm mũ thường cũng ít được ưu tiên để dùng vì sẽ tốn chi phí tính toán cao.
