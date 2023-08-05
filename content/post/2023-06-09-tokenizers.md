---
title: Tokenizers
date: 2023-06-09
tags: ["tokenizer", "word tokenization", "character tokenization", "subword tokenization", "BPE", "Byte Pair Encoding"]
---

<style>
    .singleImg {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .textSingleImg {
        text-align: center;
    }
    .twoImgBlock {
        display: flex;
        just-content: center;
    }
    .twoImg {
        width: 400px;
        height: 400px;
    }
</style>

- [Introduction](#introduction)
- [Token, Tokenizer ??](#token-tokenizer-)
- [Tokenization](#tokenization)
    - [Word Tokenization](#word-tokenization)
    - [Character Tokenization](#character-tokenization)
    - [SubWord tokenization](#subword-tokenization)
        - [Byte Pair Encoding (BPE)](#byte-pair-encoding-bpe)
        - [WordPiece tokenization](#wordpiece-tokenization)


# Introduction

Có khá là nhiều bài viết nghiên cứu sâu về việc cách thức “máy móc” đọc văn bản kiểu gì ?

Những bài viết nghiên cứu về ngôn ngữ và chuyên sâu về học thuật hơn khá nhiều, mình nghĩ nếu có thể thì các bạn có thể tìm đọc thêm.

Trong bài viết này, mình sẽ chỉ nói theo một cách, gọi là theo ý hiểu của mình, và mình nghĩ nó ở tầng trên của phần kiến thức nhiều hơn. Nào ta bắt đầu nha.

<img src="/img/tokenization/0.jpg" class="singleImg">
<p class="textSingleImg">Hình 0<p>

# Token, Tokenizer ??

Nếu cho một câu ví dụ như  “Who are you ?”, với chúng ta khi nhìn vào, ở một khía cạnh tiếp cận một cách cơ bản thì, chúng ta sẽ phân tích câu, để xem đây là một câu “trần thuật, hỏi đáp, ..v.v” hay là gì, rồi là phân tích thành phần câu như kiểu “đây là câu hỏi nơi trốn hay hỏi cái gì, là câu phủ định hay khẳng định, chủ ngữ đâu, vị ngữ đâu …v.v. Kiểu kiểu vậy”.

Yuppp, đương nhiên để có thể phân tích cũng như phân tích một cách chính xác thì chúng ta cần phải học mới biết. Tuy nhiên, một điều cơ bản nhất trước khi phân tích được câu đó, là việc ta cần phải tách được câu thành các thành phần nhỏ hơn.

Như kiểu, à cái “Who” này nè nó được gọi là 1 từ, cái “are” và “you” cũng vậy và 3 từ đó ghép lại thành câu ban đầu đấy. Hay “w”, ”h”, ”o”, ”a”, ”r”, ”e”, ”y”, “o”, ”u” là đống chữ cái mà ghép chúng lại với nhau cũng tạo thành câu ban đầu đấy ….v.v. Or các kiểu tách thành phần khác nhau dạng vậy.

Thì đối với máy móc, ta định nghĩa các thành phần như vậy để cấu thành lên 1 câu là “token”, “token” có thể đại diện cho “chữ cái (char)”, “từ (word)”, “từ con (subword)” …v.v..

Tokenizer, sẽ là các công cụ, phương pháp để giúp tách một câu thành các thành phần nhỏ hơn theo một hình thức đơn vị nào đó.

# Tokenization

Trong phần này mình sẽ điểm qua một số phương pháp để phân tích và tách câu mà mình có biết, tìm hiểu và đọc được.

Bao gồm: Word, Character, Subword, WordPiece, SentencePiece tokenization.

Về việc thực hiện tokenize, có rất nhiều các thức để có thể tiếp cận:

* Tokenize bằng từng chữ cái, coi mỗi chữ cái là 1 token
* Tokenize theo “dấu khoảng trắng” và coi mỗi token cách nhau bởi một “khoảng trắng”
* Tokenize theo các dấu câu
* Tokenize theo các Rules để đưa ra các token phù hợp
* Tokenize bằng model, ta có thể xây dựng một số model chuyên biệt dành riêng cho việc tokenize để tối ưu và phù hợp cho bài toán hoặc ngôn ngữ nào đó
* …v.v.

<img src="/img/tokenization/1.jpg" class="singleImg">
<p class="textSingleImg">Hình 1<p>

Mỗi một cách thực hiện tokenize sẽ đem lại ưu nhược điểm cũng như sự phù hợp cho từng loại bài toán, hay ngôn ngữ khác nhau.

Ví dụ, nếu ta xét ngôn ngữ là Tiếng Anh, thì ta có thể thấy việc tokenize bằng dấu “khoảng trắng” thôi cũng mang lại hiệu quả khá cao và phù hợp.

Tuy nhiên, cùng cách tiếp cận đó mà sử dụng cho các ngôn ngữ như Tiếng Trung, Tiếng Nhật hay thậm trí là Tiếng Việt mình thì độ hiệu quả đã giảm đi rất nhiều rồi.

Như tiếng Việt chúng ta, từ “chúng ta” có thể được ghép bằng 2 từ đơn “chúng” và “ta” và cách nhau bởi dấu khoảng trắng, nhưng thực tế trong hầu hết các ngữ cảnh ta mong muốn đấy là 1 từ “chúng_ta” hơn là 2 từ đơn “chúng”, “ta”.

## Word Tokenization

Work tokenization, tách từ, là một dạng tách câu thành từng token, với mỗi token là 1 từ được định nghĩa theo ngôn ngữ đó.

<img src="/img/tokenization/2.jpg" class="singleImg">
<p class="textSingleImg">Hình 2<p>

Ở trên là một số ví dụ mà ta có thể tiếp cận để thực hiện tách từ, có thể là theo khoảng trắng, dấu câu, hay bất cứ cách tiếp cận nào làm cho các câu được tách thành các từ.

Tuy nhiên, cho dù cách nào đi nữa thì việc tiếp cận thực hiện “tách từ” sẽ luôn tồn đọng một số vấn đề cốt lõi:

1. Cần bộ từ điển lớn: Khi làm việc với cách tiếp cận “tách từ” thì, cta chỉ học được những từ có trong bộ từ điển khi thực hiện huấn luyện mà thôi.

    Tất cả, các từ khác không có trong dữ liệu huấn luyện sẽ không thể biết và học được, thường chúng sẽ được đánh dấu bằng 1 token nào đó (ví dụ “UNK”) để nhận biết rằng đấy là 1 từ lạ.

    Khi đó, nếu bộ dữ liệu của cta không đủ lớn để phủ hết toàn bộ từ vựng (thực ra rất khó để phủ hết) thì sẽ luôn tồn tại các từ  “UNK”.

    Ví dụ, trong dữ liệu huấn luyện ta học được từ “không”, tuy nhiên, thực tế ta lại gặp từ “khôngggggg” thì khi đó từ “khôngggggg” này đã là một “UNK”, do vậy, mà sẽ gần như là không thể nào có thể phủ hết được toàn bộ các từ.

2.	Các từ viết tắt: đối với work token, thì từ viết tắt cũng là một trở ngại lớn, khi mà không chỉ các từ viết tắt thông dụng mà sau đó là hàng tấn từ teencode được sinh ra hằng ngày mà không bao giờ có giới hạn. Thì word token cũng k thể sử lý được


## Character Tokenization

Nhìn nhận thấy một số nhược điểm lớn như vậy, thì một cách tiếp cận khác cũng được hướng tới là việc tiếp cận theo “ký tự”.

Vì lượng ký tự như các dấu câu, chữ cái, các ký tự đặc biệt dùng thường xuyên, hay thậm trí cả không được dùng thường xuyên thì số lượng các ký tự đó cũng không đến mức quá lớn.

Và chỉ cần từng ấy ký tự có thể cấu thành lên toàn bộ từ vựng, câu văn, đoạn văn, văn bản bất kỳ. Điều đó làm cho nhược điểm về “số lượng từ vựng”, “các từ viết tắt” của cách tiếp cận word token không còn nữa.

Tuy nhiên, nó cũng tồn tại một số nhược điểm có thể kể đến:

<img src="/img/tokenization/3.jpg" class="singleImg">
<p class="textSingleImg">Hình 3<p>

1.	Thiếu ngữ nghĩa: không giống như các “từ”, mỗi từ vốn đều có nghĩa riêng của nó, trong khi đó, mỗi ký tự hầu hết không mang ý nghĩa nào cả. Do vậy, quá trình học biểu diễn, không có gì đảm bảo rằng các ký tự sẽ mang bất kỳ ý nghĩa nào. Trong khi đó, với mỗi “từ” trong từng ngữ cảnh khác nhau có thể mang ý nghĩa khác nhau.

2.	Tăng chi phí tính toán: Đương nhiên rồi, thay vì mỗi token là 1 từ, thì giờ đây 1 từ sẽ gồm nhiều token ký tự, do vậy, 1 câu được biểu diễn theo character sẽ gồm nhiều token hơn so với biểu diễn theo word. Dẫn đến, vector biểu diễn lớn => tính toán sẽ tốn chi phí hơn.

## Subword Tokenization

Sau khi kinh qua các vấn đề của các cách tiếp cận trên. Lại mong muốn rằng, có thể tiếp cận việc phân tách thành phần câu sao cho không bị mất từ, hay nói một cách khác là không muốn thấy các “UNK” token trong khi không muốn phải xây dựng một từ điển khổng lồ hay vô hạn các token.

Đồng thời cũng không muốn tiếp cận theo dạng từng ký tự như “character token” để tránh bị mất (intent meaning) ngữ nghĩa vốn có của cấp độ từ.

Khi đó Subword tokenization được sinh ra, với ý tưởng chính là việc “ta có thể xây dựng các token mới bằng những token đã biết”.

Ví dụ, từ “chúng ta” có thể được ghép từ 2 từ đơn “chúng” và “ta” vậy là với 2 từ ta vẫn có thể biểu diễn cho 3 từ là “chúng” “ta” và “chúng ta” hay tiếng anh như “unfortunately” sẽ được ghép bởi các token “un” + “for” + “tun” + “ate” + “ly”.

Do vậy, Subword tokenization, sẽ thực hiện việc xây dựng bộ từ điển gồm các “subword”. Điều này làm cho cách tiếp cận này, không phải tốn nhiều bộ nhớ như cách tiếp cận “word tokenization” mà vẫn cho phép chúng ta có thể cover được hết toàn bộ “word”.

Nếu lượng vocab của bạn nhỏ thì có thể bạn sẽ cần nhiều token để biểu diễn 1 “UNK Word”, còn nếu lượng vocab lớn thì có thể sẽ cần ít token hơn.

Ví dụ, như BERT, với cách tiếp cận theo 1 biến thể theo dạng “Subword” thì vocab_size ~ 30k token để có thể cho BERT cover hết các từ và có một biểu diễn hợp lý.

Subword tokenization là một cách tiếp cận phân tách câu với việc xây dựng bộ từ điển với các subword và sẽ biểu diễn các “unk work” bằng các subword đó.

Thế vào ví dụ thực tế đi:

Ta có đoạn văn **“Miền Bắc Việt Nam có bốn mùa là mùa Xuân mùa Hạ mùa Thu và mùa Đông còn miền nam thì không”**. Để cho dễ nhìn ta sẽ coi dấu “_” là dấu “ “.

Khi đó ta có:

<img src="/img/tokenization/4.jpg" class="singleImg">
<p class="textSingleImg">Hình 4<p>


ta có bảng chứa các từ như trên. Khi đó, sẽ có hàng tá cách để có thể phân tách 1 từ thành các “subword”

Ví dụ như từ: “Miền” có thể thành “Mi” + “ền”, “M” + “iền”, “Miề” + “n” …v.v.. Do vậy, ta sẽ chọn các “subword” như thế nào.

Khi đó, có các cách tiếp cận, phương pháp khác nhau để tính toán và lựa chọn ra các “subword” phù hợp.

Sau đây, chúng ta sẽ đi qua 1 số loại trong số các phương pháp tiếp cận là Byte pair encoding, probabilistic subword tokenization, unigram subword tokenization, wordPiece và cuối cùng là sentencePiece.

### Byte Pair Encoding (BPE)

BPE ban đầu được sử dụng trong việc nén dữ liệu “data compression” bằng việc tìm các cặp byte chung của dữ liệu.

Tuy nhiên, phương pháp tiếp cận này cũng đã được sử dụng rất phổ biến trong NLP đóng vai trò giúp chúng ta có thể thực hiện tokenize một cách hiệu quả.

Mục tiêu chính của BPE, là việc làm sao để có thể biểu diễn được toàn bộ dữ liệu với số lượng token trong vocab ít nhất có thể.

BPE sẽ thực hiện tách toàn bộ câu thành các ký tự đơn lẻ (Như m.n đã biết thì nếu biểu diễn ở dạng token là “char” thì đã hoàn toàn có thể biểu diễn được tất cả các từ với 1 lượng vocab nhỏ, tuy nhiên khi biểu diễn sẽ làm cho câu được biểu diễn “dài hơn” là so với cách biểu diễn dạng “word token”).

Sau đó, BPE thực hiện ghép 2 token gần nhau có tần suất xuất hiện cùng nhau cao nhất lại với nhau, và thêm vào trong từ điển (điều này sẽ làm tăng kích thước của từ điển lên, nhưng đổi lại số lượng token khi biểu diễn 1 câu sẽ giảm đi, vì 2 char liền nhau xuất hiện nhiều nhất đã được gộp lại thành 1 thay vì 2 như trước).

Lặp lại cho đến khi đạt được giới hạn số lượng token hoặc lặp qua 1 số lượng lần nhất định nào đó mà bạn cảm thấy OK.

Vì dựa vào cặp token xuất hiện nhiều nhất ở mỗi bước lặp để quyết định ghép đôi token, nên có thể nói việc ghép đôi của BPE như đang sử dụng thuật toán “Tham lam” (Greedy Algorithm) để quyết định ghép đôi vậy.

Việc này sẽ trade-off giữa kích thước của vocab và độ dài của câu khi biểu diễn sang dạng token.

Tuy nhiên, không phải 1 câu được biểu diễn càng ngắn càng tốt, vì nếu lượng dữ liệu mà lớn thì mà một câu dài ngoằng chỉ cần biểu diễn bằng 3-4 token chẳng hạn thì khi đó vocab_size đang cực lớn

=> hiệu suất của model sẽ giảm khi model phải học quá nhiều token khác nhau, chưa kể tới việc khi biểu diễn 1 câu dài chỉ bằng 3-4 token tức 1 token có thể là gộp của rất nhiều word khác nhau

=> điều này cũng dẫn tới việc mất đi khá nhiều ngữ nghĩa nội tại mà mỗi từ đã có sẵn.

Ngược lại, nếu vocab quá nhỏ thì sẽ bị rơi vào trường hợp giống cách tiếp cận “char token”

Quay trở lại ví dụ ban đầu, khi đó BPE sẽ thực hiện như thế nào, vẫn là câu trên ta thực hiện đếm số lượng các từ.

<img src="/img/tokenization/4.jpg" class="singleImg">
<p class="textSingleImg">Hình 5<p>


<img src="/img/tokenization/5.jpg" class="singleImg">
<p class="textSingleImg">Hình 6<p>


Ở đây thực chất bảng “Word” ở trên cũng không thực sự cần thiết, mình thêm vào để tiện cho việc đếm tần suất thôi.

Phân tách theo dạng char token thì ta có:

**vocab = ( _ , m , n , a , ù , h , i , t , c , ề , u , ô , g , b , v , à , ó , ố , l , ệ , x , â , ạ , đ , ì , k , ắ , ò )**

Ta nhận thấy 2 token “m” và “ù” liền nhau và cùng xuất hiện với nhau 5 lần, ta thực hiện ghép đôi lại với nhau, khi đó ta thêm token mới là “mù” vào trong vocab

=> **vocab = ( _ , m , n , a , ù , h , i , t , c , ề , u , ô , g , b , v , à , ó , ố , l , ệ , x , â , ạ , đ , ì , k , ắ , ò )**

Sau đó, lặp lại việc thực hiện đếm và ghép đôi token, sau khoảng 14 lần ta sẽ có thêm 1 danh sách các token mới ngoại trừ danh sách char token ban đầu như sau:

<img src="/img/tokenization/6.jpg" class="singleImg">
<p class="textSingleImg">Hình 7<p>


=> khi đó **Vocab = ( _ , m , n , a , ù , h , i , t , c , ề , u , ô , g , b , v , à , ó , ố , l , ệ , x , â , ạ , đ , ì , k , ắ , ò, mù , mùa , mùa_ , n_ , mi , miề , miền , miền_ , na , nam , nam_ , à_ , th , ôn , ông )**

mình có thể tiến hành lặp tiếp và làm mới vocab tuy nhiên, đây chỉ là ví dụ về cách thức hoạt động của BPE thôi

trong thực thế copus sẽ lớn hơn nhiều chứ không phải chỉ 1 câu như này và số lần lặp để ghép đôi cũng sẽ do từng bài toán hoặc các rule mà người dùng đặt ra có thể sẽ khác nhau sao cho có 1 vocab phù hợp.

Lúc này, ví dụ như từ “mùa_” mình chỉ cần dùng 1 token để biểu diễn thay vì 4, hay từ “và_” mình chỉ cần 2 token thay vì 3.

Hoặc khi chẳng may gặp 1 từ mới như từ “thông” thì mình vẫn có thể biểu diễn được bằng 2 token “th” và “ông” or từ “mùng” có thể biểu diễn bằng 3 token là “mù”, “n” và “g”…v.v.

Ta có thể thấy, trong quá trình thực hiện tokenize, BPE áp dụng rule để merge các cặp token lại với nhau để tạo ra token mới và cứ lần lượt như vậy.

=> Do đó, sau khi train xong BPE, để thực hiện tách từ cho 1 câu. BPE sẽ thực hiện tách toàn bộ thành các ký tự riêng lẻ, sau đó thực hiện merge theo thứ tự đã thêm vào vocab trong quá trình học ở trên.

### WordPiece Tokenization

WordPiece được giới thiệu trong bài báo về BERT của google năm 2018.

Giống với BPE:

	* WordPiece cũng xây dựng bộ từ điển bắt đầu từ nhỏ -> lớn.
	* WordPiece cũng thực hiện merge token giống BPE

Khác với BPE:

	* WordPiece thực hiện chọn các cặp token để merge bằng cách tính “điểm” cho các cặp token thay vì tính “tần suất xuất hiện” như BPE

“điểm” cho từng cặp token được tính bằng công thức:

<img src="/img/tokenization/7.jpg" class="singleImg">
<p class="textSingleImg">Hình 8<p>


<ul>
    <li>WordPiece xác định các subwords bằng cách thêm tiền tố “##” ở đầu các subwords.</li>
        <p>Ví dụ: từ “Mùa” được tách thành “M, ##ù, ##a”</p>
    <li>WordPiece cũng chỉ cần lưu lại vocab cuối cùng mà không cần phải nhớ “merge rules” giống như BPE.</li>
</ul>

Bằng việc tính score như trên, thì sẽ làm cho thuật toán khi thực hiện merge các cặp token sẽ ưu tiên các cặp có “tần suất xuất hiện cùng nhau” cao nhưng “tần suất xuất hiện đơn lẻ” lại thấp.

<img src="/img/tokenization/5.jpg" class="singleImg">
<p class="textSingleImg">Hình 9<p>


Phân tách thành dạng character ta có:

**vocab = ( _ , m , ##ù ,  ##a , ##i , ##ề , ##n , n , ##m , ##b , ##ắ , ##c , v , ##ệ , ##t , c , ##ó , ##ố , l , ##à , c , ##u , ##â , h , ##ạ , t , ##h , đ , ##ô , ##g , ##ì , k , ##ò )**

Note: như đã đề cập ban đầu, wordPiece sử dụng prefix “##” để thể hiện rằng đấy là 1 thành phần của 1 từ (mà không phải thành phần bắt đầu của từ).

* khi thực hiện merge token, ta thực hiện lược bỏ prefix “##” ở giữa 2 token khi merge đi

Tiếp theo, thực hiện tính SCORE để quyết định merge token, ta thấy có 3 cặp token có điểm bằng nhau và bằng 1.0 là “##ắ##c , ##ệ##t , h##ạ”. Chọn cái nào cũng được, giả sử chọn ghép cặp token “##ắ” và “##c” ta có “##ắc”.

=> **vocab = ( _ , m , ##ù ,  ##a , ##i , ##ề , ##n , n , ##m , ##b , ##ắ , ##c , v , ##ệ , ##t , c , ##ó , ##ố , l , ##à , c , ##u , ##â , h , ##ạ , t , ##h , đ , ##ô , ##g , ##ì , k , ##ò , ##ắc )**

Cứ tiếp tục lặp lại, các cặp token lần lượt được ghép thêm là “##ệt”, “hạ”, “bắc”, “bố”, “có”, “cò”, “là”, “xu”, “xuâ”, “và”, “đô”, “##iề”, “vi”, “việt”.

Ở đây mình dừng ở lần lặp thứ 15 để làm demo, còn trong thực tế thì ta còn có thể lặp được hơn nhiều nữa đến khi nào đạt được Vocab_size mong muốn thì thôi.

**=> vocab = ( _ , m , ##ù ,  ##a , ##i , ##ề , ##n , n , ##m , ##b , ##ắ , ##c , v , ##ệ , ##t , c , ##ó , ##ố , l , ##à , c , ##u , ##â , h , ##ạ , t , ##h , đ , ##ô , ##g , ##ì , k , ##ò , ##ệt, hạ, bắc, bố, có, cò, là, xu, xuâ, và, đô, ##iề, vi, việt )**

Khi đó, câu ban đầu:

**“miền bắc việt nam có bốn mùa là mùa Xuân mùa hạ mùa thu và mùa đông còn miền nam thì không”**

sẽ được tách thành

**“m##iề##n bắc việt n##a##m có bố##n m##ù##a là m##ù##a xuâ##n m##ù##a hạ m##ù##a t##h##u và m##ù##a đô##n##g cò##n m##iề##n n##a##m t##h##ì k##h##ô##n##g”**
