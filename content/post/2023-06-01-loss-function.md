---
title: Loss, Cost and Objective Function
date: 2023-06-01
tags: ["loss function", "cost function", "objective funtion"]
---

<style>
.textSingleImg {
    text-align: center;
}
</style>

- [Introduction](#introduction)
- [The different between loss, cost and objective function](#the-different-between-loss-cost-and-objective-function)
- [Some Note](#some-note)


# Introduction

Các khái niệm xoay quanh loss, cost và object function là những thứ, mình tin chắc là bạn nên biết khi tìm hiểu về các kiến thức liên quan tới “machine learning”

Cơ bản thì, loss function là 1 hàm số cho phép bạn đo lường sự sai khác giữa kết quả mà model dự đoán ra với kết quả thực. Với mong muốn thiết kế các model sao cho sự sai khác này là nhỏ nhất có thể (tức kết quả được dự đoán ra từ model giống với kết quả thực nhất có thể). Vậy, nếu sự sai khác này là lớn, hay kết quả được đoán ra từ model sai so với kết quả thực. Thì giá trị của loss function sẽ lớn (càng sai nhiều càng lớn) và ngược lại.

Tuy nhiên, sẽ có một số khái niệm khác mà bạn sẽ thấy không chỉ có “loss function” mà còn có “cost và objective function”.

Đừng để bị nhầm lẫn giữa các khái niệm đó nhé. Với những gì mình hiểu và mình có tìm hiểu được thì chúng thực sự khác nhau đấy. Mặc dù, thực tế thì sẽ có nhiều case mà chúng sẽ có ý nghĩa như nhau, hoặc đôi khi m.n dùng chung 1 tên cho toàn bộ khái niệm trên. Nhưng tốt nhất ta vẫn nên phân biệt được chúng thì tốt nhất.

# The different between loss, cost and objective function

Loss function đo lường sự sai khác giữa giá trị mà model dự đoán ra với giá trị thực của chúng. Nó sẽ đánh giá hiệu suất của mô hình học máy có tốt hay không trên 1 “ví dụ đào tạo cụ thể” hoặc trên từng “ví dụ đào tạo đơn lẻ”. Sẽ có nhiều cách để định nghĩa 1 loss function, vì điều này còn phụ thuộc vào mục đích của từng bài toán khác nhau. Tuy nhiên, về cơ bản thì nó thường là 1 function định nghĩa trên “điểm dữ liệu”, “kết quả dự đoán của model” và “nhãn (kết quả thực)” và đôi khi sẽ là thêm các hệ số để phạt hoặc chuẩn hoá.

<!-- {{< figure link="../../themes/beautifulhugo/static/img/loss/1.jpg" >}} -->
<img src="/img/loss/1.jpg">

<p class="textSingleImg"><b>Hinh 1</b></p>

Với Hình 1, ở trên, \\( x_i \\) là một điểm dữ liệu và \\(y_i\\) là nhãn tương ứng, \\(f(x_i|\theta)\\) là đầu ra của mô hình với điểm dữ liệu \\(x_i\\), trong đó \\(\theta\\) là tham số của mô hình, sẽ học trong quá trình huấn luyện.

Có thể thấy, 2 loss function ở trên đều sẽ tính sự sai khác giữa “kết quả dự đoán của mô hình” và “nhãn của dữ liệu”, tuy nhiên với 2 bài toán khác nhau thì cách định nghĩa hàm loss cũng sẽ khác nhau.

Cost function thường sẽ mang nghĩa tổng quát hơn so với loss function. Cost function thường được tính bằng “tổng” hoặc “trung bình” của các loss trên “toàn bộ” tập dữ liệu huấn luyện.

Với mỗi từng điểm dữ liệu, ta thực hiện tính giá trị của loss function cho điểm dữ liệu đó, sau đó “tổng” hoặc “trung bình” của toàn bộ giá trị của các loss lại ta sẽ có giá trị của cost.

Tối thiểu hoá cost function trong quá trình training sẽ giúp tối ưu các tham số của mô hình.

<img src="/img/loss/2.jpg">
<p class="textSingleImg"><b>Hinh 2</b></p>

Như hình 2, có thể thấy “Mean Squared Error (MSE)” là cost function cho bài toán Linear regression. Nó sẽ là trung bình của các giá trị loss trên “toàn bộ” tập dữ liệu huấn luyện.

Objective function có thể nói là mang tính tổng quát nhất. Một bài toán có thể có 1 hoặc nhiều các thành phần khác nhau, có thể phải cần kết hợp nhiều hơn 1 loss hay nhiều hơn 1 cost function để giải quyết bài toán. Thì objective function, là “hàm mục tiêu”, mục tiêu của bài toán.

Objective function sẽ hướng dẫn để giúp cho thuật toán tìm được các giá trị tối ưu cho các tham số của mô hình.

Objective function có thể nói là thuật ngữ mang tính tổng quát nhất cho bất kỳ function nào được dùng để tối ưu trong quá trình training.

<img src="/img/loss/3.jpg">
<p class="textSingleImg"><b>Hinh 3</b></p>

Ở hình 3, objective function là tổng của 2 Cost function khác nhau. Khi đó, Objective function sẽ là hàm hướng dẫn mô hình học để tối ưu các tham số, và nó cũng là hàm số cuối cùng để ta có thể theo dõi và quan sát tình trạng của mô hình đó (đã hội tụ hay chưa, có gặp vấn đề gì không …v.v)

# Some Note

Ở nhiều ngữ cảnh, ta có thể đâu đó nhìn thấy các khái niệm trên được dùng chồng chéo lên nhau. Sẽ có nhiều tài liệu hoặc nhiều chỗ sẽ coi luôn, loss và cost function là một, và dùng loss cho cả 2. Hoặc, khi mà bài toán chỉ cần 1 cost function thì objective = cost function mà loss lại được dùng thay cho cost.\
=> cuối cùng chỉ dùng mỗi thuật ngữ “loss function” :D. Và nhiều khi nó cũng thành thói quen luôn. Nhưng không sao cả, ta cứ hiểu sự khác biệt giữa chúng thôi. Còn lại thì miễn sao ta vẫn hiểu được là được :D.
