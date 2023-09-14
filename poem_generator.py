# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import torch
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
# 导入tokenizer
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
# model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
# 两个训练好的模型
model1 = torch.load('models/model1.model').to('cpu')
model2 = torch.load('models/model2.model').to('cpu')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(825, 358)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 240, 211, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(40, 150, 311, 71))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(500, 60, 241, 211))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(400, 50, 20, 231))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(410, 40, 331, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(410, 280, 331, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 40, 301, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 120, 311, 16))
        self.label_3.setObjectName("label_3")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(210, 110, 87, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(310, 110, 87, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "古诗生成器"))
        self.pushButton.setText(_translate("MainWindow", "生成古诗"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">秋去冬来</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "请输入1-5个字，并选择生成类型，点击按钮生成古诗"))
        self.label_3.setText(_translate("MainWindow", "可以使用如下测试文本："))
        self.comboBox.setItemText(0, _translate("MainWindow", "五言绝句"))
        self.comboBox.setItemText(1, _translate("MainWindow", "七言绝句"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "model1"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "model2"))
        self.pushButton.clicked.connect(self.clickButton)

    def data_process(self, y_pred):
        '''
        由于我们不允许生成不是中文字的字符，所以我们需要进行以下转换，
        首先现需要提剔除分隔符，未知符号，padding符号。
        还需要剔除标点符号。
        将以上字符全部设为负无穷，即不可能被采样到。
        最后进行前50大的softmax采样
        '''
        # 第50大的值,以此为分界线,小于该值的全部赋值为负无穷
        topk_value = torch.topk(y_pred, 50).values
        topk_value = topk_value[:, -1].unsqueeze(dim=1)
        # 赋值将不是前50大的值都设置为负无穷，保证不会被采样到
        y_pred = y_pred.masked_fill(y_pred < topk_value, -float('inf'))
        # 不允许写特殊符号，这里的sep，unk，pad分别代表分隔符，未知符号，padding符号（一般是none）
        # 将其都设置为负无穷，保证不会被采样到
        y_pred[:, tokenizer.sep_token_id] = -float('inf')
        y_pred[:, tokenizer.unk_token_id] = -float('inf')
        y_pred[:, tokenizer.pad_token_id] = -float('inf')
        # 同时也不允许生成标点符号
        for i in '，。!（《》）？；：、,.!?」abcdefghijklmn':
            y_pred[:, tokenizer.get_vocab()[i]] = -float('inf')
        # 根据softmax概率采样,无放回
        # 不直接采用最大值的原因是容易陷入局部最小值使得生成的文章流畅性不行，同时也千篇一律
        # 这点在实践中验证过了
        y_pred = y_pred.softmax(dim=1)
        y_pred = y_pred.multinomial(num_samples=1)
        return y_pred

    def generate_poem(self, model, text, row, col, flag):
        '''
        该为生成函数，工作原理为：给定每句话的长度，和生成几句，按照要求进行生成
        并且不允许生成奇怪的字符与字幕
        在到达规定字数后强制添加标点符号
        '''

        # 首先将text文本tokenize为可以输入进模型的tensor x
        x = tokenizer.batch_encode_plus([text], return_tensors='pt')
        x['input_ids'] = x['input_ids'][:, :-1]
        x['attention_mask'] = torch.ones_like(x['input_ids'])
        x['token_type_ids'] = torch.zeros_like(x['input_ids'])
        x['labels'] = x['input_ids'].clone()

        #因为两个数据集使用的是不同的标点符号，所以在生成时也需要使用不同的标点符号
        dots1 = ['.', '。']
        dots2 = [',', '，']
        for i in range(row * col + 4 - len(text)):
            with torch.no_grad():
                y_pred = model(**x)

            # 得到输出y_pred，取输出的最后一个，就是我们新生成的汉字
            y_pred = y_pred['logits']
            y_pred = y_pred[:, -1]
            # 剔除不需要的字符保证生成的是中文字，并采样
            y_pred = self.data_process(y_pred)

            # 在每句的之后强制添加标点符号，单数句添加逗号，复数句句号
            pos_now = (i + 1 + len(text)) / (col + 1)
            if pos_now % 2 == 0:
                y_pred[:, 0] = tokenizer.get_vocab()[dots1[flag]]
            elif pos_now % 2 == 1:
                y_pred[:, 0] = tokenizer.get_vocab()[dots2[flag]]

            x['input_ids'] = torch.cat([x['input_ids'], y_pred], dim=1)
            x['attention_mask'] = torch.ones_like(x['input_ids'])
            x['token_type_ids'] = torch.zeros_like(x['input_ids'])
            x['labels'] = x['input_ids'].clone()

        # 输出
        poem = tokenizer.decode(x['input_ids'][0])[6:]
        # 进行换行
        poem = poem.replace(dots1[flag]+' ', "。\n")
        poem = poem.replace(dots2[flag]+' ', "，\n")
        self.label.setText(poem)

    def clickButton(self):
        self.label.setText("生成中")
        text = self.textEdit.toPlainText()
        type = self.comboBox.currentText()
        type2 = self.comboBox_2.currentText()
        # 这里是选择使用的是哪种模型
        if type2 == 'model1':
            model = model1
            flag = 0
        else:
            model = model2
            flag = 1
        model.eval()
        # 选择生成什么类型
        if type == '五言绝句':
            self.generate_poem(model, text, row=4, col=5, flag=flag)
        else:
            self.generate_poem(model, text, row=4, col=7, flag=flag)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())