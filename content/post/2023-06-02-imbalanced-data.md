---
title: Imbalanced Data
date: 2023-06-02
tags: ["imbalanced data", "imbalance dataset", "data imbalance"]
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
}
.twoImg {
    display: inline;
    width: 300px;
    height: 300px;
    margin-left: 30px;
}
</style>

# Giới Thiệu

Trong thực tế thường khá là khác so với khi ta học và thực hành về machine learning hay AI ở trên trường lớp hay các khóa học trên mạng, nơi mà có các dữ liệu mẫu rất đẹp, và thuận lợi, công việc của chúng ta chỉ cần là cầm luôn đống dữ liệu đó và đi xây dựng các model học máy, có chăng thì sẽ chỉ thêm một số bước tiền xử lý (preprocessing) nhỏ trước khi dùng.

Tuy nhiên, trong thực tế thì khác, dữ liệu thực tế khi chúng ta làm việc thường sẽ là một đống dữ liệu với sự mất cân bằng giữa các nhãn với nhau. Thường sẽ có sự mất cân bằng nhẹ mà các nhãn có sự chênh lệch nhau về dữ liệu nhưng không phải quá nhiều, thì khi đó các model học máy vẫn có thể phát huy sức mạnh mà không bị ảnh huởng quá nhiều.

Nhưng cũng có một phần các bài toán mà lượng dữ liệu chênh lệch nhau lớn dẫn tới mô hình học máy xây dựng không còn hoạt động tốt dẫn tới sự ngộ nhận chất lượng mô hình và thường rất tệ trong việc dự đoán các lớp thiểu số.

Trong bài viết này chúng ta sẽ kinh qua một số phương pháp cũng như hướng giải quyết khi gặp các bài toán có độ chênh lệch dữ liệu giữa các nhãn cao (Imbalanced dataset) mà mình biết và cùng nhau bổ sung, thảo luận về chúng.

# Imbalanced Dataset

Thế thì Imbalanced Dataset là gì ?

Có thể hiểu Imbalanced Dataset là sự mất cân bằng dữ liệu, khi mà có một hoặc nhiều nhãn trong bộ dữ liệu có lượng dữ liệu lớn hoặc nhỏ hơn nhiều so với số còn lại thì được gọi là Imbalanced Dataset (hay mất cân bằng dữ liệu).

<img class="twoImg" src="/coffee/img/Imbalanced_dataset/1.png">
<img class="twoImg" src="/coffee/img/Imbalanced_dataset/2.png"><br/>
<div class="textTwoImg">
    <p>Hình 1: Imbalanced dataset phân loại 2 lớp</p>
    <p>Hình 2: Imbalanced dataset phân loại nhiều lớp</p>
</div>


Thì khi Imbalanced Dataset xảy ra thì có một số các ảnh hưởng tiêu tực tới việc xây dựng mô hình học máy. Khi mất cân bằng dữ liệu, sẽ có những nhãn (label) mà mô hình được nhìn thấy nhiều điểm dữ liệu mẫu hơn và có những nhãn được nhìn thấy ít điểm dữ liệu mẫu hơn.

Khi đó mô hình sẽ thiên về việc học tính năng cho các nhãn có nhiều dữ liệu hơn dẫn tới việc ngộ nhận chất lượng mô hình. Vì chỉ cần mô hình dự đoán tất cả đều là lớp có nhiều điểm dữ liệu thì độ chính xác mô hình (accuracy metric) đã ở mức cực kỳ cao rồi.

Ví dụ trong phân loại 2 lớp có tỉ lệ dữ liệu cho (chó và mèo) là (9:1) thì chỉ cần đoán tất cả là chó thì cũng đã có được 90% độ chính xác rồi.

Các bài toán bị mất cân bằng dữ liệu luôn là những bài toán khó nhằn theo mình thấy là vậy, dưới đây chúng ta sẽ đi qua một số các hướng tiếp cận khác nhau mà mình biết cho bài toán Imbalanced Dataset.

# Hướng Tiếp Cận

Duới đây mình sẽ trình bày một số hướng tiếp cận cho bài toán "mất cân bằng dữ liệu" mà mình biết, vì mình cũng chỉ biết, mọi người có thể cho thêm đánh giá cũng như các cách tiếp cận khác để mọi người cùng thảo luận nhé ạ.

## Làm cân bằng lại dữ liệu

Theo đúng nghĩa đen, thì ta sẽ làm cân bằng lại "lượng" dữ liệu để cho dữ liệu sẽ cân bằng nhau hơn.

### Over sampling
Over sampling là cách gọi chung của các phương pháp giúp tăng kích thước của các mẫu dữ liệu thiểu số lên để cân bằng lại so với các lớp dữ liệu có kích thước lớn hơn. Trong các phương pháp đó có thể kể đến 2 phương pháp chính:

* Giả sử một cách ngây thơ là các dữ liệu mới (nếu có) nó sẽ giống với các dữ liệu đã sẵn có. Từ đó ta sẽ thực hiện random sample các dữ liệu ở các lớp thiểu số và đắp ngược vào để có thể tăng kích thước lớp dữ liệu và cân bằng với các lớp đa số còn lại.

* Ta sẽ làm tăng kích thước dữ liệu ở các lớp thiểu số bằng cách sử dụng một số các thuật toán, phương pháp augument data để tạo ra các điểm dữ liệu giả lập và làm cân bằng dữ liệu hơn so với các lớp dữ liệu đa số còn lại.

### Under sampling

Under sampling, ngược lại với phương pháp "Over sampling" ở trên là phương pháp cắt giảm lượng dữ liệu ở các lớp dữ liệu đa số để cân bằng dữ liệu so với các lớp dữ liệu thiểu số. Có thể thực hiện cắt giảm bằng cách random sample ngẫu nhiên các mẫu sẽ được cắt giảm.

Như mình đã nói ở phần đầu, việc chênh lệch dữ liệu giữa các nhãn nếu không quá lớn thì cũng sẽ không ảnh hưởng quá nhiều tới mô hình học máy và mô hình học máy vẫn sẽ phát huy tốt sức mạnh của nó.

Do vậy, cả khi thực hiện over sampling hay under sampling ta cũng không cần quá cứng nhắc phải đưa tỉ lệ dữ liệu về ngang bằng nhau mà có thể thử nghiệm các tỉ lệ cao dần xuống thấp (ví dụ khi có 2 nhãn, ta có thể thử nghiệm từ tỉ lệ 70:30, 60:40 rồi tới 50:50).

### Thu thập thêm dữ liệu

Điều này đôi khi trong thực tế có hơi vô nghĩa vì khó để có thể thu thập thêm các mẫu dữ liệu cho các lớp thiểu số, vì nếu có thể thì đã không xảy ra việc Imbalanced Dataset =)).

Tuy nhiên, trong trường hợp nếu có thể thì cũng nên xem xét kỹ lại xem có khả năng nào cho việc thu thập thêm các mẫu dự liệu hay không và cân nhắc việc đó, nếu được thì đây cũng là một cách tốt để làm cân bằng dữ liệu cực tốt mà =)).


## Thay đổi phương pháp đánh giá mô hình

Imbalanced Dataset sẽ làm cho một số cách thức đánh giá mô hình của chúng ta không còn đúng nữa.

Ví dụ nếu chúng ta sử dụng accuracy metric để đánh giá cho mô hình phân loại, thì khi đó như đã đề cập từ trước nếu tỉ lệ dữ liệu là 9:1 thì chỉ cần model dự đoán tất cả thuộc lớp có mẫu dữ liệu đa số thì đã có được độ chính xác lên tới 90% rồi, kể cả model dự đoán sai hoàn toàn ở lớp có mẫu dữ liệu là thiểu số.

Do vậy phương pháp đanh giá đã không còn hợp lý nữa.

Thay vào đó, ta cần đánh giá độ chính xác trên từng label riêng biệt hay đưa ra confussion matrix để có cái nhìn tổng quan hơn về mức độ tốt của model.

## Phạt mô hình khi học

Ngoài các phương pháp liên quan tới việc scale lại dữ liệu, thì phạt model trong khi training cũng là một phương pháp hiệu quả.

Ý tưởng chính là việc khi training ta có thể gắn thêm một bộ trọng số để phạt đối với các lớp dữ liệu có kích thước lớn để sao cho cùng với một lần cập nhật trọng số thì giá trị khi cập nhật lớp dữ liệu kích thước lớn sẽ không mang lại nhiều giá trị bằng lớp dữ liệu kích thước nhỏ.

Ví dụ, để dễ hiểu, nếu như ta có bài toán phân loại 2 lớp chó và mèo tỉ lệ dữ liệu (8:2), ta có bộ trọng số phạt khi cập nhật là class_weight =  (0.1, 1). Tức là, với mỗi lần cập nhật thì phải cập nhật 10 lần lớp dữ liệu "chó" mới có thể bằng 1 lần cập nhật đối với lớp dữ liệu "mèo". Điều này giúp cho model học một cách công bằng hơn đối với các lớp dữ liệu.

## Chia nhỏ bài toán

Trong thực tế, bài toán Imbalanced Dataset ta có thể gặp ở cả các bài toán phân loại 2 lớp lẫn phân loại nhiều lớp. Trong đó, sẽ có những bài toán phân loại nhiều lớp mà ở đó đột nhiên có một số lớp có lượng dữ liệu trội lên hẳn so với các lớp dữ liệu còn lại.

<img class="singleImg" src="/coffee/img/Imbalanced_dataset/2.png">
<p class="textSingleImg">Hình 3: Imbalanced Dataset với multi label</p>

Khi đó, có một cách tiếp cận hiệu quả mà ta có thể xem xét là việc chia nhỏ bài toán ra thành các bài toán con.

<img class="twoImg" src="/coffee/img/Imbalanced_dataset/3.png">
<img class="twoImg" src="/coffee/img/Imbalanced_dataset/4.png">
<div class="textTwoImg">
<p>Hình 4: Chia nhỏ bài toán thành bài toán tổng quan phân loại 2 lớp là chó và không phải chó </p>
<p>Hình 5: Đối với lớp không phải là chó sẽ thực hiện phân loại nhiều nhãn như bình thường</p>
</div>

Như hình trên, ta có thể chia thành 2 bài toán

1. Phân loại chó và không phải chó (tức phân loại lớp chó và lớp không phải chó sẽ rơi vào các lớp còn lại).

2. Sau đó, từ các lớp dữ liệu không phải chó. Nơi mà dữ liệu không còn bị Imbalanced nữa, ta sẽ thực hiện phân loại nhiều lớp ở đây như bình thường.

Việc chia nhỏ bài toán tới mức nào sẽ phụ thuộc vào việc dữ liệu bị Imbalanced đến đâu, mà ta có thể lặp đi lặp lại.
