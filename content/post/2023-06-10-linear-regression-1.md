---
title: Linear, Ridge and Lasso Regression
date: 2023-06-10
tags: ["linear regression", "ridge regression", "lasso regression", "l1 normalization", "l2 normalization"]
---

<style>
.textSingleImg {
  text-align: center;
}
.textTwoImg {
    display: flex;
    margin-left: 10px;
    flex-direction: row;
    justify-content: space-around;

}
.singleImg {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 300px;
    height: 300px;
}
.twoImg {
    display: inline;
    width: 300px;
    height: 300px;
    margin-left: 30px;
}
</style>

Như đã biết thì “học máy” (machine learning) là việc kết hợp các giả thuyết về dữ liệu và công cụ toán học để đưa ra thông tin, quy luật của dữ liệu.

Các giả thuyết đặt ra thông qua insight trong quá trình EDA (Exploratory Data Analysis) để phù hợp với bài toán.

Trong bài viết này, ta sẽ tiếp cận “Linear Regression” và các “biến thể” với góc tiếp cận từ các giả thuyết.

# Linear Regression

Thường khi nhắc đến Linear Regression, ta thường xét đến “simple linear regression” một loại “linear regression model” đơn giản nhất với chỉ 1 “biến độc lập (independent variable)” và 1 “biến phụ thuộc (dependent variable)”.

Với giả thuyết đầu ra (dependent variable) phụ thuộc vào tổ hợp tuyến tính của các thuộc tính đầu vào.

Ví dụ, trong mặt phẳng 2 chiều, 1 đường thưởng y = ax + b.

* x: biến độc lập
* y: biến phụ thuộc
    phụ thuộc vào tổ hợp tuyến tính của x (ax + b)

<img class="singleImg" src="/img/linear_regression/1.jpg">

Có một định lý, là “định lý giới hạn trung tâm” (central limit theorem), ở đây định lý nói rằng “phân phối xác suất của biến ngẫu nhiên giản lược sẽ hội tụ về một phân phối chuẩn”.

Giả thuyết đưa ra là, đầu ra Y sẽ tuân theo phân phối chuẩn phụ thuộc vào tổ hợp tuyến tính của các thuộc tính đầu vào.

Ta có:

\\[ y_n \sim \mathcal{N}(x_na, \sigma) \\]

<img class="singleImg" src="/img/linear_regression/2.jpg">

Linear Regression giả định rằng phương sai sẽ là một hằng số và không đổi dữ liệu dạng này được gọi là “Homoskedastic data”.

Tuy nhiên, Trong thực tế thì phương sai thường “không” phải là hằng số, nó thường biến thiên, và dạng dữ liệu này gọi là “Heteroskedastic data”, đôi khi có thể là một hàm của giá trị trung bình. Khi giá trị trung bình tăng => variance tăng theo.

<img class="twoImg" src="/img/linear_regression/3.jpg">
<img class="twoImg" src="/img/linear_regression/4.jpg">


Ta có:

* \\( \\{x_n, y_n\\}, 0 < n < N \\) là các cặp điểm dữ liệu đã biết
* Giả thuyết \\( y_n \sim \mathcal{N}(x_na, \sigma) \\)
* Giả thuyết \\( \sigma \\) cố định.

=> Áp dụng MAP, ta cần tối đa khả năng xảy ra:

\\[ a_\* = argmax \sum^n log P(y_n | x_n; a) \\]
\\[ \ \ \ \ \ \ \ \ = argmax \sum^n log N(y_n; x_na, \sigma ) \\]
\\[ \ \ \ \ \ \ \ \ = argmax \sum^n log(\[{1 \over \sigma \sqrt{2 \pi}}e^{-{1 \over 2}({y_n - x_na \over \sigma})^2}\]) \\]
\\[ \ \ \ \ \ \ \ \ = argmax \sum^n (\[log{1 \over \sigma \sqrt{2 \pi}} - {{1 \over 2}({y_n - x_na \over \sigma})^2}\]) \\]

Vì \\( \sigma \\) cố định, là hằng số

\\( \Rightarrow log{1 \over \sigma \sqrt{2 \pi}} \\) là hằng số và \\( {1 \over 2 \sigma} \\) là hằng số

\\[ a_\* = argmax - \sum^n (y_n - x_na)^2 \\]
\\[ \Rightarrow a_\* = argmin \sum^n (y_n - x_na)^2 \\]

Tức là, vì:
* Biến độc lập x, và biến phụ thuộc y, là ta đã đều biết vì chúng là training dataset.
* Trong khi đó, a là biến duy nhất ta chưa biết và cần tìm
* Mà giả thuyết đặt ra là y tuân theo phân phối chuẩn phụ thuộc vào tổ hợp tuyến tính của các feature đầu vào là (x, a)

=> ta cần tìm \\( a_* \\) để cùng với **x**, cho ra đầu ra **y** thuộc phân phối chuẩn, nhất có thể.

# Ridge Regression

Ridge Regression đưa thêm giả thuyết về vector a với việc, vector a cũng sẽ thuộc phân phối chuẩn với \\( \mu = 0 \\) và \\( \sigma = \sqrt{1 \over \lambda} \\)

\\[ a \sim \mathcal{N}(0; \sqrt{1 \over \lambda}) \\]

Khi đó, ta có thể biểu diễn như sau:

<img class="singleImg" src="/img/linear_regression/5.jpg">

\\[ y_n \sim \mathcal{N}(x_na, \sigma) \\]
\\[ a \sim \mathcal{N}(0, \sqrt{1 \over \lambda}) \\]

=> Áp dụng MAP, ta cần tối đa hóa khả năng xảy ra:

\\[ a_* = argmax \sum^n log[P(a|x_n, y_n, \lambda)] \\]
\\[ \ \ \ \ \ \ \ \ = argmax \sum^n [logP(y_n|x_n, a) + logP(a|\lambda)] \\]
\\[ \ \ \ \ \ \ \ \ = argmax \sum^n [log \mathcal{N} (y_n; x_na, \sigma) + log \mathcal{N} (a; 0; \sqrt{1 \over \lambda})] \\]
\\[ \ \ \ \ \ \ \ \ = argmax \sum^n [log(\[{1 \over \sigma \sqrt{2 \pi}}e^{-{1 \over 2}({y_n - x_na \over \sigma})^2}\]) + log({1 \over \sqrt{2 \pi \over \lambda}}e^{-\lambda a^2 \over 2})] \\]
\\[ \ \ \ \ \ \ \ \ = argmax \sum^n [[log{1 \over \sigma \sqrt{2 \pi}} - {1 \over 2 \sigma^2 (y_n - x_na)^2}] + (log \sqrt{\lambda \over 2 \pi} - {1 \over 2}\lambda a^2)] \\]

Vì giả thuyết đưa ra "Variance" là cố định:

\\[ \Rightarrow a_* = argmax \sum^n [- (y_n - x_na)^2 - \lambda a^2] \\]
\\[ \ \ \ \ \ \ \ \ = argmax - \sum^n [(y_n - x_na)^2 + \lambda a^2] \\]
\\[ \Rightarrow a_* = argmin [\sum^n (y_n - x_na)^2 + \lambda \sum^n a^2] \\]

Khi đó, Ridge Regression sẽ là “Linear Regression” cộng thêm 1 biểu thức đằng sau là \\( \lambda a^2 \\).

Ta thấy, về cơ bản Ridge Regression là việc đưa ra thêm các giả thuyết có các biến đầu vào (cụ thể ở đây là đưa thêm giả thuyết cho **a**).

Thay vì phải tìm **a** trên toàn bộ miền, thì Ridge Regression đưa ra thêm giả thuyết **a** sẽ thuộc phân phối chuẩn

=> điều này sẽ thu hẹp miền tìm kiếm của **a** từ rất lớn về thành theo phân phối chuẩn.

Nếu \\( \lambda \rightarrow 0 \\):

* Nếu nhìn vào công thức thì ta có thể thấy luôn, khi đó công thức sẽ giống với “Linear Regression” ở phần trước.

* Nếu xét về mặt lý thuyết hơn, thì như giả thuyết được đặt ra \\( a \sim \mathcal{N}(0, \sqrt{1 \over \lambda}) \\) thì khi \\( \lambda \rightarrow 0 \\) => variance \\( \sigma \rightarrow \infty \\).\
    Khi đó, khoảng giá trị mà **a** có thể nhận được trải rộng tới vô cực -> miền tìm kiếm của **a** vẫn như cũ và không có thay đổi gì.

Nếu \\( \lambda \rightarrow \infty \\):

* variance \\( \sigma \rightarrow 0 \\). Khi đó, khoảng giá trị mà **a** có thể nhận được sẽ rất nhỏ => dẫn tới việc dễ bị underfitting.

# Lasso Regression

Giống với Ridge, nhưng thay vì đưa ra giả thuyết với \\( a \sim \mathcal{N} \\), thì Lasso đưa ra giả thuyết **a** tuân theo phân phối Laplance với \\( \mu = 0 \\) và \\( b = {1 \over \lambda} \\)

\\[ a \sim \mathcal{L}(0, {1 \over \lambda}) \\]

Với giả thuyết đưa ra từ ban đầu là "Variance" cố định. Biến đổi như Ridge Regression ở phần trước ta cần tối ưu khả năng:

\\[ a_* = argmin [\sum^n (y_n - x_na)^2 + \lambda \sum^n |a|] \\]

# Thảo Luận

Học máy là việc kết hợp các giả thuyết phù hợp với bài toán và công cụ toán học để giải quyết.

Linear Regression là việc đưa ra giả thuyết cho outcome và mối quan hệ của các biến.

Ridge và Lasso Regression cũng là kết quả của việc đưa ra thêm giả thuyết cho các trọng số của Linear Regression.

Nếu tiếp cận các bài toán theo dạng "grapphical model" sẽ giúp ta có nhiều góc nhìn để giải quyết bài toán hơn.

