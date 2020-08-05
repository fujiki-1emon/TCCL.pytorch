# TCCL.pytorch
## Temporal Cycle-Consistency Learning
### 書誌情報
- Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, Andrew Zisserman
- Google Brain and DeepMind
- CVPR2019

### 問題
- 動画の時系列位置合わせの問題を扱う。
  - 動画内の行動を認識するような課題と比較して、動画の時系列的な発展の関係を扱うこと比較的少ないため、研究の価値がある。
  - self-supervised な方法で時系列位置合わせが可能であるため、フレームごとの対応を取るような大量の supervision を必要としない。
  - self-supervised に時系列位置合わせをした後、1つの動画の他のモダリティのデータ (e.g. 音) やアノテーションラベルを、位置合わせした他の動画に伝搬させることができる。また、時系列の異常検知に応用することができる。

### 手法・要点
手法のポイントは、動画の画像フレームの時系列的な対応関係が取れるような embedding space を学習することである。
画像から訓練済みモデル等 一般的な CNN を使って抽出した画像特徴量に基づいて embedding に変換する。
異なる動画の画像フレーム (s_i, t_j) から求めた embedding の系列 (u_i, v_j)に対して、その時系列的な位置の対応を取る損失を取る。

![スクリーンショット 2020-08-05 11 45 57](https://user-images.githubusercontent.com/8359397/89366155-43cac700-d711-11ea-80a8-ac71db5750c5.png)

より具体的には、u の soft nearest neighbor (v_tilda) を求め、その soft nearest neighbor を cycle-back した時に、元の u と一致する時系列のインデックスを持つようにする損失を取る。
損失には、Cycle-back Classification と Cycle-back Regression の2つが提案されている。Cycle-back Classification は cycle-back した時の時系列インデックスを分類によって位置合わせする損失である。しかし、分類では時系列位置の遠近が考慮されないため、ターゲット位置を中心とした分布を考えて時系列のインデックスを回帰する Cycle-back Regression が提案されている。

![スクリーンショット 2020-08-05 11 43 05](https://user-images.githubusercontent.com/8359397/89366000-dae34f00-d710-11ea-8bd8-a96498e4d0ae.png)

以上の図は、原著論文から引用した。

## Implementation & Results
- 以上の Temporal Cycle-Consistency Learning を実現する最小限の実装を行なった。
  - データセットに、容器に液体を注ぐ動作を収めた動画データセット (`pouring`) を使用した。
  - また、Cycle-back Regression が最も良い結果であったという報告に基づいて、その損失のみを扱った。
- 学習した embedding によって最近傍を求めて位置合わせをした結果を以下に掲載する。
  - 500 イテレーションを学習したモデルは、対応するフレームをある程度求めることができているが、対応が崩れた箇所が見られる。
  - 3,500 イテレーションを学習したモデルは、上のモデルより良くフレームの対応づけを取ることができている。
- 他の動画での有効性など汎化性能を詳細に確かめることまではできておらず、その確認は課題である。

### 結果例1
- 500 イテレーション [[動画]](https://drive.google.com/file/d/1-hUpZsJjmZj5tA2zP4bvQJBmFW4igXT7/view?usp=sharing)

![20200803152325 checkpoint_00500 0_17](https://user-images.githubusercontent.com/8359397/89364418-65c24a80-d70d-11ea-94fc-51441b3e8689.gif) 

- 3,500 イテレーション [[動画]](https://drive.google.com/file/d/1rv5545Jr5zWgch4j5PKOFe6Kp0owq354/view?usp=sharing)

![20200803152325 checkpoint_03500 0_17](https://user-images.githubusercontent.com/8359397/89364480-88ecfa00-d70d-11ea-88ed-f0cfa63783d1.gif)

### 結果例2
- 500 イテレーション [[動画]](https://drive.google.com/file/d/1AAwKBwf4-f1WRf0ZZaXPXR5h2c1cAX09/view?usp=sharing)

![20200803152325 checkpoint_00500 50_67](https://user-images.githubusercontent.com/8359397/89364425-6955d180-d70d-11ea-86f2-be95e9ecc172.gif)

- 3,500 イテレーション [[動画]](https://drive.google.com/file/d/1_J0wTGfguIqbs9nAyM3Uxrz3dBgzWvQg/view?usp=sharing)

![20200803152325 checkpoint_03500 50_67](https://user-images.githubusercontent.com/8359397/89364495-8e4a4480-d70d-11ea-8b7b-87bdedb254ad.gif)

## Usage
- Dowload the pouring dataset (https://drive.google.com/file/d/1GVyv1oPv7-a08zKx_aANikgtaSJrSvEt/view?usp=sharing) and put it in the `data/` directory.
- `cd src/` and run the program like,

```
$ python main.py --normalize_indices --num_frames 20 --batch_size 4 
```
If you have more GPU memory (~30,000MiB), 
```
$ python main.py --normalize_indices --num_frames 32 --batch_size 8 (--weight_decay 0.)
```
Found that the latter settings resulted in a better/stable learning by a few experiments. Might need more hyper parameter tuning with the implementation.

## Acknowledgement
- Temporal Cycle-Consistency Learning [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.pdf)
- Pytorch+Tensorflowのちゃんぽんコードのすゝめ（tfdsでpytorchをブーストさせる話） [[Qiita]](https://qiita.com/namahoge/items/71be06c8fe37e88909c9#comments)
