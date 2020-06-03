# dnn_samples
## 参考文献
### cifar10
- CIFAR-10でaccuracy95%--CNNで精度を上げるテクニック--<br>
https://qiita.com/yy1003/items/c590d1a26918e4abe512
  - functional apiでのcifar10の書き方を参考にした
 

 - 現場で使える！TensorFlow開発入門 Kerasによる深層学習モデル構築手法
   - cifar10のデータインポートや正規化など参考 4章・5章
   
 
 ### 蒸留 Distillation
 - Kerasによる知識の蒸留 (knowledge distillation) ~TGS(kaggle)~<br>
 https://qiita.com/M_Hiro/items/0ba24788c78540046bcd
   - 一番参考にした記事
 
 
 - keras-knowledge-distillation<br>
 https://github.com/tripdancer0916/keras-knowledge-distillation
   - 蒸留実装で参考にした
 
 
 - kerasで知識の蒸留(Distillation)<br>
 http://tripdancer0916.hatenablog.com/entry/2018/10/14/keras%E3%81%A7%E7%9F%A5%E8%AD%98%E3%81%AE%E8%92%B8%E7%95%99_1
   - 説明と実装あり。上の蒸留手法について説明あり
 
 
 - Knowledge-Distillation-Keras-1<br>
 https://github.com/Incremental-Learning/Knowledge-Distillation-Keras-1/blob/master/Knowledge_Distillation_Notebook.ipynb
   - cifar10に蒸留手法を実装したもの。自分で実装を再現ができなかった
 
 
- モデルの蒸留を実装し freesound2019 コンペで検証してみた。<br>
https://kaeru-nantoka.hatenablog.com/entry/2019/07/20/020920
  - かえるるるさんのPytorchによる手法。特に参考にしていない<br>
  https://github.com/moskomule/distillation.pytorch/blob/master/hinton/utils.py
 
 
 - Knowledge Distillation with NN + RankGauss<br>
 https://www.kaggle.com/mathormad/knowledge-distillation-with-nn-rankgauss
   - 特に参考にしていない
 
 
 - kdtf<br>
 https://github.com/DushyantaDhyani/kdtf
   - tfでの蒸留手法。特に参考にしていない


- U-Net Compression Using Knowledge Distillation<br>
https://karttikeya.github.io/pdf/unet_report.pdf
  - Unetに蒸留手法を適用した論文


- On Compressing U-net Using Knowledge Distillation<br>
https://arxiv.org/pdf/1812.00249.pdf
  - Unetに蒸留手法を適用した論文
 
 ### U-Net
 - Keras U-Net starter - LB 0.277<br>
 https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
 
 
 - TensorFlowでUNetを構築する<br>
https://pyhaya.hatenablog.com/entry/2019/05/12/132727
   - 1系で書かれている


- Tensorflowを使ってUNetを試す Version 2<br>
https://pyhaya.hatenablog.com/entry/2019/08/18/193915
  - 2系で書かれている


- UNet — Line by Line Explanation<br>
https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
  - 割と詳しく書かれている


 
 ### その他
 - Jupyter lab / notebookで argparseそのままで実行する方法<br>
 https://qiita.com/uenonuenon/items/09fa620426b4c5d4acf9
 
 
 
 - 普通のpython実行ファイル(argparseを含むファイル)をJupyter notebookで実行するときのメモ書き<br>
 https://qiita.com/LittleWat/items/6e56857e1f97c842b261
