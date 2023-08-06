---
title: The Rest of Transformer
date: 2023-06-16
tags: ["Transformer", "Residual", "Feed Forward"]
---

- [Giới Thiệu](#giới-thiệu)
- [Residual Connections And Feed Forward Layer](#residual-connections-and-feed-forward-layer)
- [Một Số Nhận Định Và Các Nghiên Cứu Liên Quan](#một-số-nhận-định-và-các-nghiên-cứu-liên-quan)
  - [Cắt Tỉa Multi-Head Attention Với Encoder Attention](#cắt-tỉa-multi-head-attention-với-encoder-attention)
  - [Multi-Head Cực Kỳ Quan Trong Đối Với Cross Attention (Encoder-Decoder Attention)](#multi-head-cực-kỳ-quan-trong-đối-với-cross-attention-encoder-decoder-attention)
  - [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

<style>
  #imgResidual {
    width: 700px;
    height: 300px;
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  #imgFFW {
    width: 300px;
    height: 500px;
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  .imgTitle {
    text-align: center;
  }
</style>

# Giới Thiệu

Ở các phần trước đó ta đã đi qua các phần kiến thức trọng yếu trong kiến trúc Transformer.

Trong phần này, ta sẽ cùng nhau tìm hiểu nốt một số kiến thức nhỏ khác được thêm vào trong transformer như các phụ gia trong món ăn và cùng nhau đi qua một số nhận định mà mình survey được khi tìm hiểu về Transformer.

# Residual Connections And Feed Forward Layer

Vector đầu ra của Multi-Head Attention được cộng thêm với positional embedding, được gọi là residual connection.

Và sau đó đầu ra của residual connection sẽ được chuẩn hóa thông qua layer Normalization rồi được đưa qua lớp Feed Forward. Bao gồm 2 linear layer và kẹp ở giữa là một activate function RELU.

<img id="imgResidual" src="/img/Transformer/Rest/residual.png">
<p class="imgTitle">Hình 1: Residual được Normalize</p>

Residual connections và feed forward là 2 thành phần nhỏ nhưng góp phần quan trọng trong quá trình train của model.

Như đã biết, self-attention là cơ chế giúp các token nắm bắt mối quan hệ của chính nó với các token khác trong câu.

Tuy nhiên, ở đó thì self-attention không chứa các thông tin về vị trí và cho phép luồng thông tin tùy ý đi qua mạng.

Do vậy, residual connection đóng một vai trò để luôn nhắc nhở cho self-attention rằng "các token được biểu diễn theo thứ tự như vậy là nó có ý nghĩa của nó đấy nên đừng có đảo loạn các token lên nhé".

<img id="imgFFW" src="/img/Transformer/Rest/ffw.png">
<p class="imgTitle">Hình 2: Feed Forward layer</p>

Sau đó, đầu ra của residual connection được đưa vào layer feed forward với các kết nối đầy đủ để mang lại khả năng biểu diễn phong phú hơn cho các features với 2 lớp linear và kẹp ở giữa là hàm relu.

Để có thể ổn định mạng, trước khi đưa vào layer feed forward, đầu ra của residual được chuẩn hóa thông qua layer normalization.

# Một Số Nhận Định Và Các Nghiên Cứu Liên Quan

## Cắt Tỉa Multi-Head Attention Với Encoder Attention

Trong paper [3], nhóm tác giả đã chỉ ra trong Multi-Head attention có 3 loại heads chính:

* Positional heads, chú tâm vào mối quan hệ vị trí các token trong câu.
* Syntactic heads, chú tâm vào mối quan hệ cú pháp trong câu.
* Rare Words heads, chú tâm vào các từ hiếm trong câu.

Nhóm tác giả cũng đã thực hiện thử nghiệm cắt tỉa các heads trong Encoder trên tác vụ machine translation với 2 bộ dữ liệu "WMT" và "OpenSubtitles"

<img src="/img/Transformer/Rest/encoder_prune.png">
<p class="imgTitle">Hình 3: Cắt tỉa heads trong Encoder Attention [3]</p>

## Multi-Head Cực Kỳ Quan Trong Đối Với Cross Attention (Encoder-Decoder Attention)

Trong paper [4], nhóm tác giả đã chỉ ra tầm quan trọng của multi-head đối với Encoder-Decoder Attention. Nhóm tác giả thực hiện cắt giảm dần các heads trong cross attention layer trong tác vụ machine translation và nhận thấy sự sụt giảm mạnh độ chính xác của model.

<img src="/img/Transformer/Rest/cross_attention_prune.png">
<p class="imgTitle">Hình 4: Cắt tỉa heads trong Cross Attention [4]</p>

## Tài Liệu Tham Khảo

[1] <a href="https://arxiv.org/abs/1706.03762">Ashish Vaswani et al, “Attention Is All You Need”, NeurIPS 2017</a>

[2] <a href="https://theaisummer.com/self-attention/">Why multi-head self attention works: math, intuitions and 10+1 hidden insights - Nikolas Adaloglou</a>

[3] <a href="https://arxiv.org/abs/1905.09418">Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. arXiv preprint arXiv:1905.09418.</a>

[4] <a href="https://arxiv.org/abs/1905.10650">Michel, P., Levy, O., & Neubig, G. (2019). Are sixteen heads really better than one?. arXiv preprint arXiv:1905.10650.</a>
