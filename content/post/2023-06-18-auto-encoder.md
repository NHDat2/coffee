---
title: AutoEncoder Model
date: 2023-06-18
tags: ["AE", "autoencoder", "AE model"]
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

AutoEncoder truyền thống là một dạng của Neural-Network được thiết kế để có khả năng học biểu diễn một cách hiệu quả của dữ liệu đầu vào mà không cần nhãn (unsupervised learning).

AutoEncoder architecture truyền thống là một trường hợp đặc biệt của “Encoder-Decoder Architecture” khi mà ở đó “input” và “output” của AutoEncoder là giống nhau.

# AutoEncoder Architecture

<img src="/img/autoencoder/1.jpg" class="singleImg">

Hình 1, ở trên là kiến trúc của AutoEncoder

gồm 3 thành phần chính:

* Encoder: Thực hiện nén và biểu diễn dữ liệu đầu vào.

* Bottelneck: Nhận đầu vào là đầu ra của Encoder => chứa các kiến thức đã được nén và biểu diễn của dữ liệu đầu vào bằng Encoder

* Decoder: Nhận đầu vào là Bottelneck, thực hiện giúp model giải nén và tái cấu trúc để trở lại thành dữ liệu ban đầu

Ví dụ với một mạng AutoEncoder đơn giản nhất:

<img src="/img/autoencoder/2.jpg" class="singleImg">

Ta có thể thấy cách thức hoạt động của AutoEncoder như sau:

<img src="/img/autoencoder/3.jpg" class="singleImg">

Ngay sau khi có được “output” ta tính toán sự khác biệt giữa “output” và “ground-truth”. Với “output” chính là việc tái tạo lại “input” ban đầu và “ground-truth” chính là “input” ban đầu luôn.

Đó là lý do, AutoEncoder là unsupervised learning.

**Loss Function**: Tuỳ thuộc vào đầu vào cũng như tính chất bài toán mà sẽ sử dụng các hàm loss khác nhau. Đối với AutoEncoder truyền thống, ta cần tính loss function là sự khác biệt giữa “output” và chính “input” của nó.

Ta có:

\\[ L_{(\theta, \varphi)} = \sum^n_i(x_i - \hat{x_i})^2 \\]
\\[ \ \ \ \ \ \ \ \ ~~~~~~~~~~~~~~~ = \sum^n_i[x_i - g_{\varphi}(f_{\theta}(x_i))]^2 \\]

# Một số nhận định và nghiên cứu liên quan

* Có thể nói bottleneck là module quan trọng nhất trong AutoEncoder Model. Nếu không có bottleneck module thì model có thể dễ dàng học bằng cách ghi nhớ các giá trị đầu vào bằng cách truyền các giá trị này qua mạng bên dưới

<img src="/img/autoencoder/4.jpg" class="singleImg">

* Bottle neck càng nhỏ, tủi ro overfitting càng thấp, tuy nhiên nếu kích thước Bottleneck quá nhỏ sẽ hạn chế khả năng lưu trữ thông tin của ảnh đầu vào, gây khó khăn cho việc giải mã ở khối Decoder

<img src="/img/autoencoder/5.jpg" class="singleImg">

* Thực tế, nếu như ta xây dựng 1 mạng AutoEncoder tuyến tính (ví dụ như mạng không dùng activate function).\
    Khi đó ta sẽ quan sát được dữ liệu sẽ giảm chiều giống giống như cách ta quan sát được khi dùng PCA.

* Một AutoEncoder Model lý tưởng là cân bằng được:
** Sự nhạy cảm với inputs để có thể tái cấu trúc lại inputs một cách chính xác
** Không quá nhạy cảm với các inputs quá đơn giản khiến model chỉ đơn giản là ghi nhớ giá trị đầu vào, hay bị overfit data

* AutoEncoder học cách nén và biểu diễn dữ liệu từ training data. Do vậy, AutoEncoder không thể nén, biểu diễn “ảnh phong cảnh” khi mà training data chỉ toàn “ảnh chữ số” được.

* Số lượng nodes ở mỗi layers sẽ giảm dần với Encoder module và ngược lại, sẽ tăng dần với Decoder module. Số lượng layers hay số lượng nodes ở Decoder và Encoder Modules không nhất thiết phải là đối xứng, ta có thể tuỳ biến tuỳ ý.
