'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_utils.py
'''
""" From GPT4:
在评估大型语言模型（LLM）生成的文本时，选择哪种相似度度量取决于您希望评估的内容类型和相似度方面。以下是关于何时使用这些度量的一些指导：

### Levenshtein距离（编辑距离）

### 余弦相似度
- 适合衡量两个句子或文档在语义层面的相似度，通常用于高维向量空间模型，如TF-IDF或词嵌入（word embeddings）表示的文本。
- 余弦相似度评估了向量化表示的文本之间的角度，这更关心文档或句子在话题或含义上的接近程度。
- 适用于评估生成文本与参考标准在整体意义上的接近性，尤其是在生成长篇文章或段落时。

在实际应用中，为了全面评估语言模型生成结果的质量，可能需要结合多种度量，或者选择在特定任务中表现良好的度量来评估：

- 对于生成的结构需要高度精确，如代码或公式，可以考虑Levenshtein距离。
- 对于关注文本关键内容的生成，如摘要或关键词提取，可以考虑Jaccard相似度。
- 对于评估表达相同意思的不同方式的生成，如文章、故事或对话，余弦相似度可能是更好的选择。

最终，无论选择哪种度量，最重要的是确保它反映了您的语言模型优化和评估目标。在某些情况下，还可能需要开发专门的评估方法来捕捉任务所需的特定文本属性。

To Run test test: it require install :

    ```
    pip install python-Levenshtei
    pip install scikit-learn
    ```

"""

from typing import List


class JiebaTokenizer:
    @classmethod
    def tokenize(cls, text: str)-> List[str]:
        import jieba
        return jieba.lcut(text)


class GenerateResultCompare:
    @staticmethod
    def tokenize(text: str, lang: str) -> List[str]:
        if lang == "zh":
            return JiebaTokenizer.tokenize(text)
        else:
            return text.split()

    def normal_similarity(self, gen_result: str, ref: str, lang: str = "en") -> float:
        raise NotImplementedError("base class not implement this function, use sub class.")


class LevenshteinCompare(GenerateResultCompare):
    """
    编辑距离测量从一个字符串转换到另一个字符串所需的最少单字符编辑操作数。每个操作包括插入、删除或替换一个字符。Python有多个库可以计算编辑距离，如python-Levenshtein。
    安装python-Levenshtein库：
    "require pip install python-Levenshtein"

    - 更适合评估生成的文本在单词层面上的精确度，例如拼写检查或OCR的输出。
    - 它对于单词的添加、删除或替换非常敏感，如果你的任务需要在字符层面上的准确度比较，编辑距离是个不错的选择。
    - 不适用于评估语义内容，因为即使两个句子语义相近但单词使用不同，编辑距离也可能很高。
    """

    def normal_similarity(self, gen_result: str, ref: str, lang: str = "en") -> float:
        """
        Args:
            gen_result:
            ref:

        Returns: [0,1.0]  0 means totally different strings, 1 means totally identical strings
        """
        import Levenshtein
        # 计算编辑距离
        distance = Levenshtein.distance(gen_result, ref)

        # 取两个字符串中较长的长度
        max_length = max(len(gen_result), len(ref))

        # 防止除以0，如果两个字符串都为空，我们可以定义他们是完全相同的
        if max_length == 0:
            raise ValueError("two empty strings")

        # 计算标准化编辑距离
        normalized_distance = distance / max_length
        return 1.0 - normalized_distance


class JaccardCompare(GenerateResultCompare):
    """
    适合评估两个句子在词汇层面的重叠情况，意在衡量它们在词集层面的相似性。它忽略了单词顺序和频率的因素，用于评估内容相似性而非形式上的精确重现。
    对于短文本或关键词提取等任务比较合适。
    """

    def normal_similarity(self, gen_result: str, ref: str, lang: str = "en") -> float:
        """
        Args:
            gen_result: genereate result
            ref: reference

        Returns: [0,1.0]  0 means totally different strings, 1 means totally identical strings
        """

        def jaccard_similarity(str1, str2):
            words_str1 = set(self.tokenize(str1, lang=lang))
            words_str2 = set(self.tokenize(str2, lang=lang))
            return len(words_str1.intersection(words_str2)) / len(words_str1.union(words_str2))

        similarity = jaccard_similarity(gen_result, ref)
        return similarity


class CosineCompare(GenerateResultCompare):
    """
    余弦相似度评估了向量化表示的文本之间的角度，这更关心文档或句子在话题或含义上的接近程度。
    适用于评估生成文本与参考标准在整体意义上的接近性，尤其是在生成长篇文章或段落时。
    """

    def normal_similarity(self, gen_result: str, ref: str, lang: str = "en") -> float:
        """
        Args:
            gen_result: genereate result
            ref: reference

        Returns: [0,1.0]  0 means totally different strings, 1 means totally identical strings
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        words_str1 = ' '.join(self.tokenize(gen_result, lang=lang))
        words_str2 = ' '.join(self.tokenize(ref, lang=lang))
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([words_str1, words_str2])

        similarity = cosine_similarity(tfidf)[0][1]
        return similarity

