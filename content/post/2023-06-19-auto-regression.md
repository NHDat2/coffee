---
title: AutoRegressive Model
date: 2023-06-19
tags: ["AR", "autoregressive", "AR model"]
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

Auto-regressive model là một loại mô hình khá quan trọng và được sử dụng trong nhiều tác vụ khác nhau.

Đồng thời nó cũng là nền móng của rất nhiều “mô hình sinh (Generative Model)”, có thể kể đến như “NADE, MADE, PixelRNN, PixelCNN, hay họ GPT, ..v.v.”.

Trong bài viết này ta sẽ cùng đi qua về sự cơ bản thế nào là một Auto-regressive model nhé mọi ngừi.

# AutoRegressive Model

Trước hết, hiểu một cách nôm na thì Autoregressive model là dạng model mà trong đó **giá trị output ở thời điểm hiện tại** sẽ phụ thuộc vào **các giá trị output ở các thời điểm trước đó** (1 hoặc nhiều giá trị trước đó).

<img src="/img/ar/1.jpg" class="singleImg">

Ở đây ta có thể thấy, Output tại thời điểm “t” sẽ phụ thuộc vào các output từ thời điểm “1 -> t-1”.

**Note**: \\( \hat{y_t} \\) là output tại thời điểm “t” của điểm dữ liệu nhé.

Ví dụ, trong dịch máy “machine translation”.

<img src="/img/ar/2.jpg" class="singleImg">

Trong ví dụ “dịch máy” ở trên, \\( \hat{y_1}, \hat{y_2}, ... , \hat{y_5} \\) là output tại các thời điểm “1, 2, 3, 4, 5”. Và output tại thời điểm “hiện tại” sẽ phụ thuộc vào output tại thời điểm trước đó ( như \\( \hat{y_4} \\) phụ thuộc vào \\( \hat{y_3} \\)).

Và cả câu “<start> I go to school” là **output** của mô hình với **điểm dữ liệu đầu vào** là “tôi đi học <end>”

Như đã biết, phân phối **Gaussian** với tham số \\( (\mu, \sigma) \\) có dạng như hình 2, phân phối **Gausian Mixture** với tham số \\( (\mu, \sigma) \\) có dạng như hình 3, phân phối **Uniform** với tham số \\( (a, b) \\) có dạng như hình 4.

<img src="/img/ar/3.jpg" class="singleImg">
<p class="textSingleImg"><b>Hinh 2, 3, 4: Đồ thị phân phối Gaussian, Gaussian Mixture và Uniform</b></p>

Phía trên là 1 số ví dụ với các phân phối quen thuộc. Nhưng có vẻ như đồ thị với bất kì hình vẽ dù cho có đơn giản hay nguệch ngoạc đến đâu cũng có **“một phân phối nào đó biểu diễn được nó với bộ tham số nhất định”**

AutoRegressive đưa ra giả thuyết rằng output của model sẽ tuân theo 1 phân phối \\( P_{\theta} \\) nào đó.

Ví dụ với Gaussian distribution: Thì với mỗi “điểm dữ liệu x” ta sẽ có output \\( \hat{y} \\) tương ứng.

<img src="/img/ar/4.jpg" class="singleImg">

Thì với giả thuyết mà AutoRegressive đưa ra, thì với mỗi “điểm dữ liệu x” ta cũng sẽ có output \\( \hat{y} \\) tương ứng

<img src="/img/ar/5.jpg" class="singleImg">

Khi đó, Autoregressive model sẽ học bộ tham số \\( \theta \\) của phân phối \\( P_{\theta} \\).

Phân phối \\( P_{\theta} \\) là phân phối của dữ liệu đầu ra.

Do vậy, trong các Generative models sử dụng Autoregressive, \\( P_{\theta} \\) sẽ phản ánh lại phân phối của chính dữ liệu đầu vào vì “output” của Generative model đó sẽ muốn tạo ra được các dữ liệu tựa tựa các dữ liệu đầu vào

=> Khi đó, nếu học được phân phối của dữ liệu đầu vào thì model có thể dễ dàng tạo ra các dữ liệu tựa dữ liệu ban đầu (vì chúng cùng thuộc 1 phân phối).

Do tính chất của Autoregressive model, với output tại thời điểm \\( \hat{y} \\) sẽ phụ thuộc vào các thời điểm trước đó, cùng với việc giả thuyết đưa ra là output của model sẽ thuộc phân phối \\( P_{\theta} \\).

Do vậy, với mỗi \\( \hat{y} \\) được dự đoán, ta cần tối ưu bộ tham số \\( \theta \\) sao cho xác suất \\( P_{\theta} \\) tại thời điểm t khi đã biết giá trị của các output tại thời điểm \\( 1 \rightarrow t-1 \\) là lớn nhất.

Ta có thể biểu diễn nó như sau:

<img src="/img/ar/6.jpg" class="singleImg">

Với việc \\( \hat{y_0} \\) được tạo ra từ input **x** (có thể đã được xử lý như encode or tương tự, “cái này tuỳ thuộc vào người thiết kế model”)

Có thể thấy công thức (*) chính là Maximum Likelihood Estimation (MLE)

# Thảo Luận

Với việc \\( \hat{y_0} \\) được tạo ra từ input **x** (có thể đã được xử lý như encode or tương tự, “cái này tuỳ thuộc vào người thiết kế model”)

Có thể thấy công thức (*) chính là Maximum Likelihood Estimation (MLE)
