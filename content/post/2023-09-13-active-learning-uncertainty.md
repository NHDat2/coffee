---
title: Active Learning - Uncertainty-based Sampling
date: 2023-09-13
tags: ["active learning", "uncertainty sampleing", "uncertainty-based sampling"]
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
    width: 500px;
    height: 500px;
    margin-left: 30px;
}
</style>

# Giới Thiệu

Active learning, là một trong những cách “learn” trong “machine learning” =)).

Thường được áp dụng cho các bài toán “supervised learning”, mà ở đó, ta gặp khó khăn với “nhãn (label)” của dữ liệu.

Ví dụ, Nếu như dữ liệu ta có quá lớn, mà ta chỉ có 1 lượng rất nhỏ nhãn của dữ liệu, or việc gán nhãn dữ liệu gặp khó khăn như “tốn rất nhiều chi phí” về cả tiền bạc lẫn công sức và thời gian, hay bất kỳ vấn đề nào khiến cho ta chỉ có được 1 phần dữ liệu “đã được gán nhãn” và phần còn lại chưa được gán nhãn thì rất nhiều => **Khi đó, Active learning thường được áp dụng vào**.

Active learning, **ý tưởng chính** là việc cho rằng các thuật toán, model machine learning có thể đạt được độ chính xác cao hơn nếu có thể tự do lựa chọn dữ liệu mà nó muốn học. Và sẽ chỉ cần gán nhãn 1 phần dữ liệu, thay vì toàn bộ dữ liệu mà vẫn có được model với hiệu suất tốt.

Nói một cách khác, tức là với lượng “dữ liệu đã gán nhãn” ít ỏi đó, ta sẽ build 1 model, sau đó, ta sẽ lựa chọn các “dữ liệu tiếp theo sẽ được gán nhãn” và thực hiện gán nhãn rồi train lại model cứ thế lặp lại, thì “nỗ lực” mà ta phải bỏ ra để gán thêm nhãn sẽ giảm đi, mà hiệu quả của model vẫn sẽ có.

<img src="/img/active_learning/1.jpg">

**Key concept, ở đây là việc lựa chọn đâu sẽ là dữ liệu tiếp theo được gán nhãn**.

Có rất nhiều cách khác nhau để xây dựng chiến lược cho việc lựa chọn dữ liệu được gán nhãn tiếp theo. Trong khuân khổ bài viết này, ta sẽ đi qua về “Uncertainty-based Sampling”

# Uncertainty-based

Uncertainty-based sampling, là một chiến lược **dựa vào “độ tin cậy (confidence)” output của ML algorithms or models** để dùng làm độ đo không chắc chắn.

Với các dữ liệu mới, model sẽ predict đưa ra output. Sau đó, sẽ sample một lượng các dữ liệu mà độ tin cậy không cao (thể hiện rằng model chưa có nhiều kiến thức về loại dữ liệu đó) để thực hiện gán nhãn cho các dữ liệu đó và đưa vào dữ liệu để train lại.

<img src="/img/active_learning/2.jpg">

Tức là, với các dữ liệu mà model khi predict cho độ tin cậy cao, (ví dụ, xác suất để xảy ra nhãn đó là 0.1 or 0.9 chẳng hạn) thì tức là, **model đã có “kiến thức” nhất định và khá tốt cho các nhãn đó rồi**, thì nếu ta gán nhãn thêm cho các dữ liệu thuộc nhãn đó thì dường như ý nghĩa mang lại không quá nhiều

Trong khi đó, nếu các dữ liệu mà model predict cho độ tin cậy thấp, (ví dụ, xác suất để xảy ra nhãn đó là 0.4, 0.5, 0.6) thì tức là, **model đang vẫn bị mập mờ về việc, dữ liệu đó thuộc nhãn nào nhỉ ?** (Vì 0.4 là 1 nhãn khác mà 0.6 là 1 nhãn khác -> độ tin cậy thấp), thì khi đó nếu ta có thể gán nhãn thêm dữ liệu thuộc các nhãn đó thì sau model sẽ dữ đoán các nhãn đó tốt hơn => mang lại hiệu quả, ý nghĩa tốt hơn so trường hợp ở trên.

    **Note**: Ta có thể thấy, việc lựa chọn “dữ liệu tiếp theo được gán nhãn” phụ thuộc vào độ tin cậy từ output của model. Do vậy, Việc thực hiện “gán nhãn” hay “độ chính xác của nhãn dữ liệu” ở đây rất quan trọng. Nếu nhãn được gán có “vấn đề” thì ngay từ đầu cái gọi là “độ tin cậy” từ output của model đã không được chính xác rồi, và tiếp tục men theo dựa vào model thì sai sẽ thêm sai.

## Uncertainty-based Sampling Methods

Có rất nhiều cách khác nhau đo lường thông tin từ dữ liệu, để quyết định xem thông tin đó **“chắc chắn, hay không chắc chắn”** để đưa ra quyết định dữ liệu đó có nên được chọn để gán nhãn tiếp hay không.

Tuy nhiên, có 2 cách mà mình hay sử dụng nhất là:

* Sử dụng luôn “xác suất” mà model đưa cho cho từng nhãn tương ứng (thường được dùng trong Binary-classification)
* Sử dụng Entropy để đo (thường được dùng trong multi-classification)

### Probability Measure

Thường xác suất đầu ra của model đã chứng minh luôn được cho model đó “chắc chắn hay không chắc chắn” về dữ liệu đầu vào luôn rồi.

Nếu ta coi ngưỡng mặc định:

* “input” thuộc lớp 0, nếu \\( P < 0.5 \\)
* “input” thuộc lớp 1, nếu \\( P \geq 0.5 \\)

Thì khi đó, output của model cho input tương ứng xoay quanh gần với giá trị 0.5 (như 0.4, 0.6, ..v.v) ta sẽ coi là model chưa chắc chắn. Đương nhiên các ngưỡng sẽ là linh động để phù hợp với bài toán và hoàn cảnh.

<img src="/img/active_learning/3.jpg">

### Entropy Measure

Một cách khác, là thay vì trực tiếp sử dụng Probability ta sẽ sử dụng độ đo “Entropy” (Nếu bạn quên, thì mình đã có 2 bài viết về “Entropy” rồi, bạn có thể xem lại)

Nếu trong binary-classification thì ta có thể dễ dàng đưa ra kết luận về việc “model có chắc chắn đối với dữ liệu đó hay là không”. Nhưng nếu trong “multi-classification” khi mà có nhiều hơn 2 lớp, ví dụ bài toán có 10 lớp, thì làm sao để ta có thể biết được model có kiến thức chắc chắn về dữ liệu đó hay không.

Ví dụ, trong bài toán 4 lớp, xác suất đầu ra của model với input tương ứng từ lớp 1 -> 4 là:

* **Case1**: input1 -> output (0.2, 0.8, 0, 0)
* **Case2**: input2 -> output (0.2, 0.6, 0.2, 0)
* **Case3**: input3 -> output (0.2, 0.4, 0.3, 0.1)

**Thì trong 3 case ở trên, case nào là model “không chắc chắn” kiến thức về input tương ứng ?**

Thực ra, như đã biết về sự “chắn chắn và không chắc chắn” của model dựa vào xác suất, ta cũng có thể hình dung ra được là, có vẻ như model mà đưa ra output xác suất của các lớp tương ứng càng gần nhau thì có vẻ sự “nhập nhằng (hay không chắc chắn)” sẽ càng cao, và ngược lại.

* Như ở Case1, lớp 2 có xác suất xảy ra rất cao là \\( P_{\(y=2|X\)} = 0.8\\) và lớp 3, 4 thì \\( P_{\(y=3\ or\ 4|X\)} = 0 \\), và lớp 1 thì \\( P_{\(y=1|X\)} = 0.2 \\) => Có vẻ model khá chắc về kiến thức với dữ liệu là input1 và đưa ra kết quả cho lớp 2 với xác suất khá cao => **model có vẻ khá chắc chắn**.

* Ở Case2, thì \\( P_{\(y=2|X\)} = 0.6 \\) đã giảm đi 1 chút vì \\( P_{\(y=3|X\)} \\) đã tăng từ 0 -> 0.2 => **model có vẻ không còn chắc chắn lắm về kiến thức đối với dữ liệu đầu vào là input2**.

* Ở Case3, thì có thể thấy xác suất xảy ra ở các lớp cũng khá khá gần nhau => **model có vẻ không tự tin, và không chắc chắn về kiến thức đối với dữ liệu đầu vào là input3**.

Ta có thể thấy, thực chất dùng “xác suất” vẫn có thể tìm được “dữ liệu” nào là dữ liệu mà model không chắc chắn cũng như nên được đưa vào để gán nhãn tiếp. Nhưng việc tính toán để đưa ra quyết định vẫn hơi mơ hồ (ở trên các số khá chẵn (0.2, 0.6, 0.8..v.v.), trong thực tế xác suất trải dài trong khoảng [0, 1] thì các số sẽ lẻ rất nhiều (0.246, 0.657, 0.454, …v.v.)) khiến việc xác định và tính toán cũng sẽ khó để đưa ra quyết định hơn.

Do vậy, thay vì dùng trực tiếp “xác suất” thì dùng “Entropy” sẽ đưa ra một con số về lượng thông tin và sẽ giúp ta dễ dàng để đo cũng như quyết định xem liệu model có “không chắc chắn” với “dữ liệu đó” hay không. Hay là “dữ liệu đó” có đáng để đưa vào làm dữ liệu được gán nhãn tiếp theo hay không.

\\[ Entropy = \sum_{i} p_i log(p_i) \\]

Như đã biết, \\( \max\(Entropy\) = log(n) \Leftrightarrow \\) đó là phân phối đều tức các xác suất bằng nhau (**Uniform Distribution**).

Vậy, nếu xét trong ví dụ trước

ta có: \\( n = 4 \Rightarrow \max\(Entropy\) = log(4) = 2 \Leftrightarrow \\) xác suất đầu ra của các lớp là bằng nhau hay \\( P_{\(y=1|X\)} = P_{\(y=2|X\)} = P_{\(y=3|X\)} = P_{\(y=4|X\)} = 0.25 \\). Thì có phải, nếu xét trên khía cạnh xác suất như nãy thì nó hoàn toàn hợp lý đúng khum, bùmmmmm =))

Và khi tính bằng Entropy sẽ đưa cho ta 1 con số cụ thể, thay vì 1 loạt số (như 4 số xác suất vừa rồi) => sẽ cho ta dễ dàng hơn trong việc đưa ra ngưỡng và quyết định.

<img src="/img/active_learning/4.jpg">

Ta có thể thấy, ở ví dụ “Uncertain” thì Entropy sẽ “cao” hơn so với ở ví dụ “Certain”.

Ta có thể hiểu, à \\( E_{uncertain}\\) cao hơn vậy tức nó mang nhiều thông tin hơn => **đáng để chọn và gán nhãn tiếp theo**. Còn \\( E_{certain}\\) khá thấp, tức mang “ít” thông tin => **Không đáng để chọn và gán nhãn tiếp theo**.

Đương nhiên, việc lựa chọn còn phụ thuộc vào ngưỡng để phù hợp với model và phù hợp với bài toán, trung bình thì ta có thể cứ lấy default ngưỡng là \\( \max(Entropy) \over 2\\)

Hay nếu ở ví dụ trên sẽ là \\( {\max(Entropy) \over 2} = {log(4) \over 2} = 1\\) => **cứ \\(Entropy > 1\\) sẽ được duyệt để chọn gán nhãn tiếp theo**.

# Thảo Luận

Còn rất nhiều phương pháp khác nhau trong “Uncertainty-based Sampling” hay các phương pháp khác trong “Active learning” mà mình chưa biết.

Trong bài viết này, mình chỉ đi vào 2 phương pháp thuộc “Uncertainty-based Sampling” mà mình biết và đã dùng rồi thôi. Nên nếu mọi người hứng thú thì m.n có thể tìm hiểu sâu hơn về “active learning”.

    Note: nhắc lại, vì đây là cách ta dựa vào mức độ chắc chắn về output của model để chọn ra tiếp các dữ liệu sẽ được gán nhãn tiếp theo

    => Do vậy, việc “gán nhãn”, “nhãn” dữ liệu rất quan trọng và cần phải chính xác để model không đi lệch hướng.
