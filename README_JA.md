相册小師、分類篩選写真。例えば、人物の自撮り写真や風景写真など。


# GallaryAI

[简体中文](README-CN.md) | [English](README-EN.md) | [日本語]

相册小師、分類篩選写真。例えば、人物の自撮り写真や風景写真など。本プロジェクトは、画像分類や管理のためのPythonスクリプト一連を含み、深層学習モデルを用いて画像ライブラリの中の画像を知能的に分類・整理することを目指しています。例えば、人物の自撮り写真や風景写真を識別・分類し、アーカイブ整理を行います。このプロジェクトは、畳み込みニューラルネットワーク（CNN）に基づいて実装され、PyTorchを主要な深層学習フレームワークとして使用しています。ここに私が訓練したモデルを提供しており、ダウンロードアドレスは以下の通りです：https://link.jscdn.cn/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBckpVdXJnZURwTHNpNE04aDRleHEyaUtKY0lFd1E_ZT1jMERFY0k.pth

### ファイル構造

```bash
├───gpu.py
├───PyTorch.py
├───cnn.py
├───ceshi.py
├───Start.ps1
├───requirement.txt
└───best_model.pth
```

プロジェクトは主に、GPUの検出スクリプト（gpu.pyとPyTorch.py）、畳み込みニューラルネットワークの実装（cnn.py）、モデルテストスクリプト（ceshi.py）、後続の自動化スタートスクリプト（Start.ps1）、および訓練済みのモデルファイル（best_model.pth）を含みます。

## 実行ガイド
本プロジェクトはPythonプログラミング言語に基づいており、主にPyTorch、Pillow（PIL）、NumPyなどの外部コードライブラリを使用して画像処理と深層学習モデルの訓練と推論を行います。プログラムの実行に使用されるPythonのバージョンは3.8.18で、Anacondaを使用してPython環境を設定することを推奨します。以下の設定プロセスは、Windows 11システムでテスト済みです。以下はWindows Terminalの指示です。

### 環境設定

```bash
conda環境を作成し、GallaryAIと命名して環境をアクティブ化
conda create -n GallaryAI python=3.8.18
conda activate GallaryAI
```


Windows:
```bash 
公式ウェブサイトに行って対応するバージョンのPyTorchをダウンロード。GPU訓練には完全版のPyTorchの手動インストールが必要
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

プログラムスクリプトを実行してPyTorchがGPUを正常に呼び出せるかテスト
python gpu.py
python PyTorch.py

外部コードライブラリをインストール
pip install -r requirements.txt
```


モデルの重みファイルはプロジェクトのcnn.pyがあるディレクトリ下のbest_model.pthに保存されます。テストスクリプトのceshi.pyはデフォルトでファイルがあるディレクトリ下のモデルファイル、つまり訓練完了後のモデルを呼び出します。異なる訓練段階のモデルのパフォーマンスを観察するには、cnn.py内でモデルの保存パスを変更できます。

モデルを再訓練する必要がある場合は、cnn.pyがあるディレクトリでこのファイルを実行できます。

曲線の閲覧
プロジェクトには訓練過程のTensorboard曲線図が含まれており、Tensorboardを使用して詳細データを閲覧できます。VSCode統合のTensorboardプラグインを使用して直接閲覧することをお勧めしますが、伝統的な方法も使用できます：

```bash
cd "所在ディレクトリ"
tensorboard --logdir=logs/
```

ブラウザでTensorboardサービスのデフォルトアドレスhttp://localhost:6006/を開くと、訓練過程のインタラクティブな曲線図を閲覧できます。
