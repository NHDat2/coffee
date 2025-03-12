---
title: Dealing With Imbalanced Datasets
date: 2023-09-15
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

Imbalanced Datasets là vấn đề mà ta thường gặp phải khi giải quyết các bài toán với dữ liệu thực tế. Và nó luôn là một trong những vấn đề khó nhằn khi giải quyết các bài toán machine learning.

Nếu lượng chênh lệch dữ liệu giữa các lớp không lớn, thì các mô hình học máy vẫn có thể xử lý và cho ra được các kết quả khả quan với hiệu suất tốt.

Nhưng cũng có một phần các bài toán mà lượng dữ liệu chênh lệch nhau lớn dẫn tới mô hình học máy xây dựng không còn hoạt động tốt dẫn tới sự ngộ nhận chất lượng mô hình và thường rất tệ trong việc dự đoán các lớp thiểu số.

Trong bài viết này chúng ta sẽ kinh qua một số phương pháp cũng như hướng giải quyết khi gặp các bài toán có độ chênh lệch dữ liệu giữa các nhãn cao (Imbalanced dataset) mà mình biết và cùng nhau bổ sung, thảo luận về chúng.

# Imbalanced Dataset

Có thể hiểu Imbalanced Dataset (hay mất cân bằng dữ liệu) là sự mất cân bằng dữ liệu, khi mà có một hoặc nhiều nhãn trong bộ dữ liệu có lượng dữ liệu lớn hoặc nhỏ hơn nhiều so với số còn lại.

<img class="singleImg" src="/img/Imbalanced_dataset/1.jpg"><br>

Thì khi Imbalanced Dataset xảy ra thì có một số các ảnh hưởng tiêu tực tới việc xây dựng mô hình học máy. Khi mất cân bằng dữ liệu, sẽ có những nhãn (label) mà mô hình được nhìn thấy nhiều điểm dữ liệu mẫu hơn và có những nhãn được nhìn thấy ít điểm dữ liệu mẫu hơn.

Khi đó, mô hình sẽ thiên về việc học tính năng cho các nhãn có nhiều dữ liệu hơn, nếu không dùng các phương pháp đánh giá mô hình phù hợp thì sẽ dẫn tới việc ngộ nhận chất lượng mô hình. Vì chỉ cần mô hình dự đoán tất cả đều là lớp có nhiều điểm dữ liệu thì độ chính xác mô hình (accuracy metric) đã ở mức cực kỳ cao rồi.

Ví dụ trong phân loại 2 lớp có tỉ lệ dữ liệu cho (chó và mèo) là (9:1) thì chỉ cần đoán tất cả là chó thì cũng đã có được 90% độ chính xác rồi.

Các bài toán bị mất cân bằng dữ liệu luôn là những bài toán khó nhằn theo mình thấy là vậy, dưới đây chúng ta sẽ đi qua một số các hướng tiếp cận khác nhau mà mình biết cho bài toán Imbalanced Dataset.

# Hướng Tiếp Cận

Duới đây mình sẽ trình bày một số hướng tiếp cận cho bài toán “mất cân bằng dữ liệu” mà mình biết, mọi người có thể cho thêm đánh giá cũng như các cách tiếp cận khác để mọi người cùng thảo luận nhé ạ.

Có một số cách tiếp cận đối với bài toán imblanced dataset có thể kể đến như:

1. Làm cân bằng lại dữ liệu
2. Thay đổi phương pháp đánh giá mô hình
3. Phạt mô hình khi học
4. Chia nhỏ bài toán
5. Ensemble learning

## 1. Làm cân bằng lại dữ liệu

Theo đúng nghĩa đen, thì ta sẽ làm cân bằng lại “lượng” dữ liệu để cho dữ liệu sẽ cân bằng nhau hơn. Các phương thức để có thể chuyển từ một phấn phối dữ liệu cho các nhãn mất cân bằng về phân phối cân bằng hơn.

<img class="singleImg" src="/img/Imbalanced_dataset/2.jpg"><br>

### 1.1 Thu thập thêm dữ liệu

Điều này đôi khi trong thực tế có hơi vô nghĩa vì khó để có thể thu thập thêm các mẫu dữ liệu cho các lớp thiểu số, vì nếu có thể thì đã không xảy ra việc Imbalanced Dataset =)).

Tuy nhiên, trong trường hợp nếu có thể thì cũng nên xem xét kỹ lại xem có khả năng nào cho việc thu thập thêm các mẫu dự liệu hay không và cân nhắc việc đó, nếu được thì đây cũng là một cách tốt để làm cân bằng dữ liệu cực tốt mà =)).

### 1.2 Resampling Techniques

Resampling là một trong những phương pháp được dùng rộng rãi khi làm việc với imbalanced datasets. Thực hiện việc:
•	**Oversampling**: Lặp lại hoặc tạo ra thêm dữ liệu mới từ “dữ liệu của các lớp nhãn có lượng dữ liệu nhỏ (lớp thiểu số)”
•	**Undersampling**: Hay loại bỏ một lượng dữ liệu nhất định từ “dữ liệu của các lớp nhãn có lượng dữ liệu lớn (lớp đa số)”
•	**Combined Resampling**: Hoặc vừa loại bỏ dữ liệu ở lớp đa số và cũng tạo ra thêm dữ liệu ở lớp thiểu số

Để nhằm mục đích giúp dữ liệu có thể giảm bớt sự mất cân bằng giữa các lớp nhãn.

#### 1.2.1 Oversampling

Over sampling là cách gọi chung của các phương pháp giúp tăng kích thước của các mẫu dữ liệu thiểu số lên để cân bằng lại so với các lớp dữ liệu có kích thước lớn hơn. Trong các phương pháp đó có thể kể đến 2 phương pháp chính:

•	Giả sử một cách ngây thơ là các dữ liệu mới (nếu có) nó sẽ giống với các dữ liệu đã sẵn có. Từ đó ta sẽ thực hiện random sample các dữ liệu ở các lớp thiểu số và đắp ngược vào để có thể tăng kích thước lớp dữ liệu và cân bằng với các lớp đa số còn lại.

•	Ta sẽ làm tăng kích thước dữ liệu ở các lớp thiểu số bằng cách sử dụng một số các thuật toán, phương pháp augument data để tạo ra các điểm dữ liệu giả lập và làm cân bằng dữ liệu hơn so với các lớp dữ liệu đa số còn lại, các điểm dữ liệu mới được tạo ra có thể là các điểm ngoại biên, liền kề, hàng xóm của các điểm dữ liệu trước đó.

<img class="singleImg" src="/img/Imbalanced_dataset/3.jpg"><br>

Các kỹ thuật phổ biến trong phương pháp này có thể kể đến là:

•	Random Oversampling
•	ADASYN (Adaptive Synthetic Sampling)
•	SMOTE (Synthetic Minority Oversampling Technique)

#### 1.2.2 Undersampling

Under sampling, ngược lại với phương pháp “Oversampling” ở trên là phương pháp cắt giảm lượng dữ liệu ở các lớp dữ liệu đa số để cân bằng dữ liệu so với các lớp dữ liệu thiểu số.

•	Ta cũng có thể 1 cách ngây thơ và đơn giản là random để cắt giảm bớt lượng dữ liệu ở các lớp dữ liệu đa số

•	Hoặc ta có thể áp dụng các phương pháp như KNN để quyết định xem các điểm dữ liệu nào sẽ được cắt giảm. (ý tưởng kiểu như các điểm dữ liệu cùng thuộc 1 lớp thì sẽ ở gần nhau, còn nếu điểm nào xa mà KNN phân loại sai thì ta sẽ coi đó là điểm nhiễu và cắt giảm các điểm đó đi)

<img class="singleImg" src="/img/Imbalanced_dataset/4.jpg"><br>

Các kỹ thuật phổ biến trong phương pháp này có thể kể đến là:

•	Random Undersampling
•	Tomek Links
•	ALLKNN ( All K-Nearest-Neighbors)
•	ENN (Edited Nearest Neighbors)

#### 1.2.3 Combined Resampling

Combined Resampling là một phương pháp kết hợp cả 2 loại trên. Thực hiện “Oversampling” đối với dữ liệu thiểu số, và “Undersampling” đối với dữ liệu đa số.

<img class="singleImg" src="/img/Imbalanced_dataset/5.jpg"><br>

Các kỹ thuật phổ biến trong phương pháp này có thể kể đến là:

•	SMOTETomek (SMOTE & Tomek)
•	SMOTEENN (SMOTE & ENN)

## 2. Thay đổi phương pháp đánh giá mô hình

Imbalanced Dataset sẽ làm cho một số cách thức đánh giá mô hình của chúng ta không còn đúng nữa.

Ví dụ nếu chúng ta sử dụng accuracy metric để đánh giá cho mô hình phân loại, thì khi đó như đã đề cập từ trước nếu tỉ lệ dữ liệu là 9:1 thì chỉ cần model dự đoán tất cả thuộc lớp có mẫu dữ liệu đa số thì đã có được độ chính xác lên tới 90% rồi, kể cả model dự đoán sai hoàn toàn ở lớp có mẫu dữ liệu là thiểu số.

Do vậy phương pháp đanh giá đã không còn hợp lý nữa.

Thay vào đó, ta cần đánh giá độ chính xác trên từng class riêng biệt hay đưa ra confussion matrix để có cái nhìn tổng quan hơn về chất lượng của model.

## 3. Phạt mô hình khi học

Ngoài các phương pháp liên quan tới việc scale lại dữ liệu, thì phạt model trong khi training cũng là một phương pháp hiệu quả.

Ý tưởng chính là việc khi training ta có thể gắn thêm một bộ trọng số để phạt đối với các lớp dữ liệu đa số để sao cho cùng với một lần cập nhật trọng số thì giá trị khi cập nhật lớp dữ liệu đa số sẽ không mang lại nhiều giá trị bằng lớp dữ liệu thiểu số.

Ví dụ, để dễ hiểu, nếu như ta có bài toán phân loại 2 lớp chó và mèo tỉ lệ dữ liệu (8:2), ta có bộ trọng số phạt khi cập nhật là class_weight = (0.1, 1). Tức là, với mỗi lần cập nhật thì phải cập nhật 10 lần lớp dữ liệu “chó” mới có thể bằng 1 lần cập nhật đối với lớp dữ liệu “mèo”. Điều này giúp cho model học một cách công bằng hơn đối với các lớp dữ liệu.

## 4. Chia nhỏ bài toán

Trong thực tế, bài toán Imbalanced Dataset ta có thể gặp ở cả các bài toán phân loại 2 lớp lẫn phân loại nhiều lớp.

Trong đó, sẽ có những bài toán phân loại nhiều lớp mà ở đó đột nhiên có một số lớp có lượng dữ liệu trội lên hẳn so với các lớp dữ liệu còn lại.

Khi đó, có một cách tiếp cận hiệu quả mà ta có thể xem xét là việc chia nhỏ bài toán ra thành các bài toán con.

<img class="singleImg" src="/img/Imbalanced_dataset/6.jpg"><br>

Như hình trên, ta có thể chia thành 2 stage:

•	Stage 1: Phân loại 2 lớp “là chó” và “không phải là chó” (tức phân loại lớp chó và lớp không phải chó sẽ rơi vào các lớp còn lại).

•	Stage 2: Sau đó, từ các “lớp dữ liệu không phải chó”. Nơi mà dữ liệu không còn bị Imbalanced nữa, ta sẽ thực hiện phân loại nhiều lớp ở đây như bình thường.

Việc chia nhỏ bài toán tới mức nào sẽ phụ thuộc vào việc dữ liệu bị Imbalanced đến đâu, mà ta có thể lặp đi lặp lại.

## 5. Ensemble Learning

Ensemble Learning là một phương pháp kết hợp nhiều thuật toán, mô hình khác nhau để đạt được thuật toán, mô hình cuối cùng có hiệu suất tốt hơn mức cho thể đạt được khi chỉ có một thuật toán hoặc một mô hình.

Đã có nhiều nghiên cứu chỉ ra rằng ensemble learning là một phương pháp rất tốt để deal with imbalanced dataset problem.

Ý tưởng đằng sau khi áp dụng ensemble vào bài toán imbalanced là khi sử dụng ensemble learning, ta sẽ kết hợp nhiều model lại với nhau, và expect các model nhỏ trong đó sẽ handle được một phần của các nhãn riêng biệt, từ đó mỗi model nhỏ làm tốt nhiệm vụ handle một phần các nhãn riêng biệt của mình và rồi kết hợp lại với nhau để đưa ra model có thể handle tốt việc bị imbalanced data.

