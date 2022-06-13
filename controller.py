from PyQt5 import QtWidgets, QtGui
from UI import Ui_MainWindow
from worker import Worker
import numpy as np

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        ''' 設定按鈕功能，生成執行緒 '''
        self.bt_switch(False)
        self.worker = Worker()
        self.ui.pushButton.clicked.connect(self.checkClicked)
        self.ui.pushButton_2.clicked.connect(self.cancelClicked)
    
    def checkClicked(self):
        ''' [確定按鈕] 執行緒運作'''
        self.bt_switch(True)
        self.worker.ticker = self.ui.lineEdit.text()
        self.worker.wait()
        self.worker.start()
        self.worker.trigger.connect(self.updateStatus)
        self.worker.error.connect(self.fixError)
        self.worker.finished.connect(self.threadFinished)
        
    def cancelClicked(self):
        ''' [取消按鈕] 執行緒停止'''
        self.worker.stop = True
        self.worker.wait()
        self.bt_switch(False)
    
    def updateStatus(self, text):
        ''' [狀態標籤] 更新狀態標籤的文字，顯示當前狀態 '''
        self.ui.label_3.setText(text)
        
    def threadFinished(self, pred):
        ''' [執行緒結束運行] 將結果以圖片和表格的形式表現在視窗中'''
        self.ui.label_2.setPixmap(QtGui.QPixmap('./temp.png'))
        self.updateTable(pred)
        self.bt_switch(False)
        self.updateStatus('結束')
        
    def updateTable(self, pred):
        ''' [表格] 顯示預測表格'''
        table = QtGui.QStandardItemModel(7,4)
        table.setHorizontalHeaderLabels(['開市','最高','最低','收市'])
        for index, p in np.ndenumerate(pred):
            table.setItem(index[0], index[1], QtGui.QStandardItem(str(p)))
        self.ui.tableView.setModel(table)
        
    def bt_switch(self, busy):
        ''' [按鈕設定] 設定按鈕是否可觸發'''
        self.ui.pushButton.setDisabled(busy)
        self.ui.pushButton_2.setDisabled(not busy)
        
    def fixError(self):
        ''' [執行緒出錯] 如果沒找到該檔股票，回傳並停止執行緒'''
        self.updateStatus('查無此代碼')
        self.worker.wait()
        self.bt_switch(False)

    def closeEvent(self, event):
        ''' [視窗結束] 關閉視窗時執行緒也要停止'''
        self.worker.stop = True