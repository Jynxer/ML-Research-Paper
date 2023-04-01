import tkinter as tk
import numpy as np
import pandas as pd
from datetime import datetime
import pywt
import yfinance as yf
from keras.layers import LSTM, Dense
from attention import Attention
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import History
from keras import metrics
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from math import isnan
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
matplotlib.use('TkAgg')

# ! Model Variables

df = pd.DataFrame()
ticker = ""
stock = yf.Ticker('')
history = pd.DataFrame()
lookAhead = 0
exposureSize = 0
trainingData = pd.DataFrame()
testingData = pd.DataFrame()
splitDate = datetime.now().date()
scaler = StandardScaler()
anotherScaler = MinMaxScaler()
X_train, y_train, X_test, y_test = [], [], [], []
todaysInputs = []
dates = pd.DataFrame()
model = Sequential()
optimiser = Adam(learning_rate=0.01)
numEpochs = 0
batchSize = 0
historyOne, historyTwo, historyThree = History(), History(), History()
totalLoss, totalValLoss = pd.DataFrame(), pd.DataFrame()
y_hat = []

# ! Methods


def UpdateVariables():
    global ticker, tickerEntry, stock, history, responseLabel, lookAhead, lookAheadEntry, exposureSize, exposureSizeEntry, numEpochs, numEpochsEntry, batchSize, batchSizeEntry
    ticker = tickerEntry.get()
    if (ticker == ''):
        responseLabel['text'] = "Please enter a ticker."
        return
    stock = yf.Ticker(ticker)
    today = datetime.now().date().strftime("%Y-%m-%d")
    history = stock.history(start="1990-01-01", end=today)
    if (history.shape[0] == 0):
        responseLabel['text'] = "Ticker not found!"
        return
    if (lookAheadEntry.get() == ''):
        responseLabel['text'] = "Please enter a number of days until prediction."
        return
    lookAhead = int(lookAheadEntry.get())
    if (exposureSizeEntry.get() == ''):
        responseLabel['text'] = "Please enter an exposure size."
        return
    exposureSize = int(exposureSizeEntry.get())
    if (numEpochsEntry.get() == ''):
        responseLabel['text'] = "Please enter a number of training epochs."
        return
    numEpochs = int(numEpochsEntry.get())
    if (batchSizeEntry.get() == ''):
        responseLabel['text'] = "Please enter a batch size."
        return
    batchSize = int(batchSizeEntry.get())
    responseLabel['text'] = "Variables updated!"
    print("Updated variables!")


def Run():
    global history, responseLabel, splitDate, exposureSize, scaler, anotherScaler, lookAhead, X_train, y_train, X_test, y_test, todaysInputs,  model, optimiser, numEpochs, batchSize, historyOne, historyTwo, historyThree, dates
    UpdateVariables()
    # Error handling
    if (history.shape[0] == 0):
        responseLabel['text'] = "Cannot train: no data loaded."
        return
    if (lookAhead == 0):
        responseLabel['text'] = "Cannot train: days to prediction not set."
        return
    if (exposureSize == 0):
        responseLabel['text'] = "Cannot train: exposure size not set."
        return
    if (numEpochs == 0):
        responseLabel['text'] = "Cannot train: number of training epochs not set."
        return
    if (batchSize == 0):
        responseLabel['text'] = "Cannot train: batch size not set."
        return

    # Build dataframe of time series
    opens = history.Open
    highs = history.High
    lows = history.Low
    closes = history.Close
    volume = history.Volume
    data = pd.concat([opens, highs, lows, closes, volume], axis=1)
    data.dropna(inplace=True, axis=0)

    # Split into training and testing sets
    testSize = 0.1
    if (data.shape[0] > 3000):
        testSize = 0.05
    trainingDataUnscaled, testingDataUnscaled = train_test_split(
        data, test_size=testSize, random_state=42, shuffle=False)
    splitDate = testingDataUnscaled.index[0].date()
    dates = testingDataUnscaled.index.values
    previousDays = trainingDataUnscaled.tail(exposureSize)
    testingDataUnscaled = pd.concat([previousDays, testingDataUnscaled])

    # Preprocessing
    scaler = StandardScaler()
    anotherScaler = MinMaxScaler()
    trainingDataStandardised = scaler.fit_transform(trainingDataUnscaled)
    trainingDataNormalised = anotherScaler.fit_transform(
        trainingDataStandardised)
    trainingDataTransformed = np.array(
        WaveletTransform(trainingDataNormalised, 3))
    testingDataStandardised = scaler.transform(testingDataUnscaled)
    testingDataNormalised = anotherScaler.transform(testingDataStandardised)
    testingDataTransformed = np.array(
        WaveletTransform(testingDataNormalised, 3))

    # Build input and output series
    offset = lookAhead-1
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(exposureSize, trainingDataNormalised.shape[0]-offset):
        X_train.append(np.concatenate(
            [trainingDataNormalised[i-exposureSize:i], trainingDataTransformed[i-exposureSize:i]], axis=1))
        y_train.append(trainingDataNormalised[i+offset, 0])
    for j in range(exposureSize, testingDataNormalised.shape[0]-offset):
        X_test.append(np.concatenate(
            [testingDataNormalised[j-exposureSize:j], testingDataTransformed[j-exposureSize:j]], axis=1))
        y_test.append(testingDataNormalised[j+offset, 0])
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(
        y_train), np.array(X_test), np.array(y_test)
    todaysInputs = np.array([np.concatenate(
        [testingDataNormalised[-exposureSize:], testingDataTransformed[-exposureSize:]], axis=1)])

    # Define and compile the model
    model = Sequential()
    model.add(LSTM(50, input_shape=(
        X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Attention(units=16))
    model.add(Dense(1))
    optimiser = Adam(learning_rate=0.01)
    model.compile(loss='mae', optimizer=optimiser, metrics=[metrics.MeanAbsoluteError(
    ), metrics.MeanSquaredError(), metrics.RootMeanSquaredError(), coefficient_of_determination])
    print(model.summary())

    # Train the model
    historyOne = model.fit(X_train, y_train, epochs=numEpochs//3,
                           batch_size=batchSize, validation_data=(X_test, y_test), verbose=1)
    if (isnan(historyOne.history['loss'][-1])):
        responseLabel['text'] = "Training failed: bad weight initialisation."
        return
    K.set_value(model.optimizer.learning_rate, 0.001)
    historyTwo = model.fit(X_train, y_train, epochs=numEpochs//3,
                           batch_size=batchSize, validation_data=(X_test, y_test), verbose=1)
    K.set_value(model.optimizer.learning_rate, 0.0001)
    historyThree = model.fit(X_train, y_train, epochs=numEpochs//3,
                             batch_size=batchSize, validation_data=(X_test, y_test), verbose=1)
    responseLabel['text'] = "Training complete!"

    # Show training report
    trainingReportWindow = tk.Tk()
    trainingReportWindow.title("Training Report")
    trainingReportTitleFrame = tk.Frame(trainingReportWindow, pady=40)
    trainingReportTitleFrame.grid(row=0, columnspan=1, sticky='EW')
    trainingReportTitle = tk.Label(trainingReportTitleFrame, text="Training Report",
                                   font=("Arial", 35, "bold"), width=35)
    trainingReportTitle.pack(pady=10)
    metricsFrame = tk.Frame(trainingReportWindow, pady=10)
    metricsFrame.grid(row=1, columnspan=1, sticky='EW')
    mae = historyThree.history['mean_absolute_error']
    mse = historyThree.history['mean_squared_error']
    rmse = historyThree.history['root_mean_squared_error']
    r2 = historyThree.history['coefficient_of_determination']
    valmae = historyThree.history['val_mean_absolute_error']
    valmse = historyThree.history['val_mean_squared_error']
    valrmse = historyThree.history['val_root_mean_squared_error']
    valr2 = historyThree.history['val_coefficient_of_determination']
    maeText = "Mean Absolute Error (MAE): " + str(mae[-1])
    mseText = "Mean Squared Error (MSE): " + str(mse[-1])
    rmseText = "Root Mean Squared Error (RMSE): " + str(rmse[-1])
    r2Text = "Coefficient of Determination (R2): " + str(r2[-1])
    valmaeText = "Mean Absolute Error (MAE): " + str(valmae[-1])
    valmseText = "Mean Squared Error (MSE): " + str(valmse[-1])
    valrmseText = "Root Mean Squared Error (RMSE): " + str(valrmse[-1])
    valr2Text = "Coefficient of Determination (R2): " + str(valr2[-1])
    trainingMetricsFrame = tk.Frame(trainingReportWindow, pady=10)
    trainingMetricsFrame.grid(row=2, columnspan=1, sticky='EW')
    trainingMetricsLabel = tk.Label(
        trainingMetricsFrame, text="Training Data Metrics:", font=("Arial", 27))
    maeLabel = tk.Label(trainingMetricsFrame, text=maeText,
                        font=("Helvetica", 20))
    mseLabel = tk.Label(trainingMetricsFrame, text=mseText,
                        font=("Helvetica", 20))
    rmseLabel = tk.Label(trainingMetricsFrame,
                         text=rmseText, font=("Helvetica", 20))
    r2Label = tk.Label(trainingMetricsFrame, text=r2Text,
                       font=("Helvetica", 20))
    trainingMetricsLabel.pack(pady=10)
    maeLabel.pack()
    mseLabel.pack()
    rmseLabel.pack()
    r2Label.pack()
    testingMetricsFrame = tk.Frame(trainingReportWindow, pady=20)
    testingMetricsFrame.grid(row=3, columnspan=1, sticky='EW')
    testingMetricsLabel = tk.Label(
        testingMetricsFrame, text="Testing Data Metrics:", font=("Arial", 27))
    valmaeLabel = tk.Label(testingMetricsFrame,
                           text=valmaeText, font=("Helvetica", 20))
    valmseLabel = tk.Label(testingMetricsFrame,
                           text=valmseText, font=("Helvetica", 20))
    valrmseLabel = tk.Label(testingMetricsFrame,
                            text=valrmseText, font=("Helvetica", 20))
    valr2Label = tk.Label(testingMetricsFrame,
                          text=valr2Text, font=("Helvetica", 20))
    testingMetricsLabel.pack(pady=10)
    valmaeLabel.pack()
    valmseLabel.pack()
    valrmseLabel.pack()
    valr2Label.pack(pady=(0, 40))
    PredictAndPlot()
    print("Finished training!")

# Save/Load model feature removed due to incompatability with the attention layer.
# def SaveModel():
#     global model, responseLabel
#     if (len(model.layers) == 0):
#         responseLabel['text'] = "Model hasn't been trained yet."
#         return
#     model.save("savedModel")
#     responseLabel['text'] = "Saved model!"
#     print("Saved model!")


# def LoadModel():
#     global model, responseLabel
#     if (not exists("./savedModel")):
#         responseLabel['text'] = "No model has been saved."
#     model = load_model('./savedModel', custom_objects={"Attention": Attention})
#     responseLabel['text'] = "Loaded model!"
#     print("Loaded model!")


def PredictAndPlot():
    global y_hat, model, X_test, y_test, y_hat, scaler, anotherScaler, ticker, splitDate, historyOne, historyTwo, historyThree, numEpochs, lookAhead, todaysInputs, dates
    # Predict
    y_hat = model.predict(X_test)

    # Undo normalisation
    y_test = (
        y_test * anotherScaler.data_range_[0]) + anotherScaler.data_min_[0]
    y_hat = (y_hat * anotherScaler.data_range_[0]) + anotherScaler.data_min_[0]

    # Undo standardisation
    y_test = (y_test * scaler.scale_[0]) + scaler.mean_[0]
    y_hat = (y_hat * scaler.scale_[0]) + scaler.mean_[0]

    # Plot prediction
    numTicks = 10
    tickDifference = dates.shape[0] // numTicks
    xTickLocations = np.arange(0, dates.shape[0], tickDifference)
    xTicks = []
    for i in xTickLocations:
        xTicks.append(pd.to_datetime(dates[i]).strftime("%Y-%m-%d"))
    predictionPlotWindow = tk.Tk()
    predictionPlotWindow.title("Performance on Testing Data")
    predictionPlotFrame = tk.Frame(predictionPlotWindow)
    predictionPlotFrame.pack()
    predictionFigure = Figure(figsize=(14, 5), dpi=100)
    predictionFigureCanvas = FigureCanvasTkAgg(
        predictionFigure, predictionPlotFrame)
    NavigationToolbar2Tk(predictionFigureCanvas, predictionPlotFrame)
    predictionPlotTitle = ticker + " Open Price Prediction " + str(lookAhead)
    if (lookAhead == 1):
        predictionPlotTitle = predictionPlotTitle + " Day in Advance"
    else:
        predictionPlotTitle = predictionPlotTitle + " Days in Advance"
    ax1 = predictionFigure.add_subplot()
    ax1.plot(y_test, color='red', label="Actual Open Price")
    ax1.plot(y_hat, color='blue', label="Predicted Open Price")
    ax1.set_title(predictionPlotTitle)
    ax1.set_xticks(xTickLocations, xTicks)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Open Price ($)")
    ax1.legend()
    predictionFigureCanvas.get_tk_widget().pack(pady=20, expand=1)

    # Plot loss and validation loss over epochs
    totalLoss = np.concatenate(
        [historyOne.history['loss'], historyTwo.history['loss'], historyThree.history['loss']])
    totalValLoss = np.concatenate(
        [historyOne.history['val_loss'], historyTwo.history['val_loss'], historyThree.history['val_loss']])

    lossPlotWindow = tk.Tk()
    lossPlotWindow.title("Training Performance over Epochs")
    lossPlotFrame = tk.Frame(lossPlotWindow)
    lossPlotFrame.pack()
    lossFigure = Figure(figsize=(7, 5), dpi=100)
    lossFigureCanvas = FigureCanvasTkAgg(lossFigure, lossPlotFrame)
    NavigationToolbar2Tk(lossFigureCanvas, lossPlotFrame)
    lossPlotTitle = ticker + " Price Prediction Performance"
    ax2 = lossFigure.add_subplot()
    ax2.plot(totalLoss, color='red')
    ax2.axvspan(0, numEpochs//3, facecolor='black',
                alpha=0.6, label="LR = 0.01")
    ax2.axvspan(numEpochs//3, 2*(numEpochs//3),
                facecolor='black', alpha=0.4, label="LR = 0.001")
    ax2.axvspan(2*(numEpochs//3), numEpochs-1,
                facecolor='black', alpha=0.2, label="LR = 0.0001")
    ax2.axis(xmin=0, xmax=numEpochs-1)
    ax2.set_xticks([2*i for i in range(numEpochs//2)],
                   [(2*j)+1 for j in range(numEpochs//2)])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error")
    ax2.set_title(lossPlotTitle)
    ax2.get_legend()
    lossFigureCanvas.get_tk_widget().pack(pady=20, expand=1)

    valLossPlotWindow = tk.Tk()
    valLossPlotWindow.title("Validation Performance over Epochs")
    valLossPlotFrame = tk.Frame(valLossPlotWindow)
    valLossPlotFrame.pack()
    valLossFigure = Figure(figsize=(7, 5), dpi=100)
    valLossFigureCanvas = FigureCanvasTkAgg(valLossFigure, valLossPlotFrame)
    NavigationToolbar2Tk(valLossFigureCanvas, valLossPlotFrame)
    valLossPlotTitle = ticker + " Price Prediction Validation Performance"
    ax3 = valLossFigure.add_subplot()
    ax3.plot(totalValLoss, color='red')
    ax3.axvspan(0, numEpochs//3, facecolor='black',
                alpha=0.6, label="LR = 0.01")
    ax3.axvspan(numEpochs//3, 2*(numEpochs//3),
                facecolor='black', alpha=0.4, label="LR = 0.001")
    ax3.axvspan(2*(numEpochs//3), numEpochs-1,
                facecolor='black', alpha=0.2, label="LR = 0.0001")
    ax3.axis(xmin=0, xmax=numEpochs-1)
    ax3.set_xticks([2*i for i in range(numEpochs//2)],
                   [(2*j)+1 for j in range(numEpochs//2)])
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Mean Absolute Error")
    ax3.set_title(valLossPlotTitle)
    ax3.get_legend()
    valLossFigureCanvas.get_tk_widget().pack(pady=20, expand=1)

    futurePricePrediction = model.predict(todaysInputs)
    currentPrice = todaysInputs[0, -1, 0]
    futurePrice = futurePricePrediction[0, 0]
    # Undo normalisation
    currentPrice = (
        currentPrice * anotherScaler.data_range_[0]) + anotherScaler.data_min_[0]
    futurePrice = (
        futurePrice * anotherScaler.data_range_[0]) + anotherScaler.data_min_[0]
    # Undo standardisation
    currentPrice = (currentPrice * scaler.scale_[0]) + scaler.mean_[0]
    futurePrice = (futurePrice * scaler.scale_[0]) + scaler.mean_[0]
    # print("Current: " + str(currentPrice))
    # print("Future: " + str(futurePrice))
    percentageChange = (futurePrice - currentPrice) / currentPrice
    actionThreshold = 0.01
    recommendation = ""
    if (abs(percentageChange) < actionThreshold):  # ! Hold
        recommendation = "hold"
    elif (currentPrice < futurePrice):  # ! Buy
        recommendation = "buy"
    elif (currentPrice > futurePrice):  # ! Sell
        recommendation = "sell"
    labelUpdate = "Today's price is $" + \
        str(currentPrice) + "\nThe predicted price in " + str(lookAhead)
    if (lookAhead == 1):
        labelUpdate = labelUpdate + " day is $"
    else:
        labelUpdate = labelUpdate + " days is $"
    labelUpdate = labelUpdate + \
        str(round(futurePrice, 2)) + \
        "\nI recommend that you " + recommendation + "!"
    responseLabel['text'] = labelUpdate
    print("Prediction made and plots shown.")


def Reset():
    global df, ticker, stock, history, lookAhead, exposureSize, date, trainingData, testingData, splitDate, scaler, anotherScaler, X_train, y_train, X_test, y_test, model, optimiser, numEpochs, batchSize, historyOne, historyTwo, historyThree, totalLoss, totalValLoss, y_hat, todaysInputs, dates
    df = pd.DataFrame()
    ticker = ""
    stock = yf.Ticker('')
    history = pd.DataFrame()
    lookAhead = 0
    exposureSize = 0
    trainingData = pd.DataFrame()
    testingData = pd.DataFrame()
    splitDate = datetime.now().date()
    scaler = StandardScaler()
    anotherScaler = MinMaxScaler()
    X_train, y_train, X_test, y_test = [], [], [], []
    todaysInputs = []
    dates = pd.DataFrame()
    model = Sequential()
    optimiser = Adam(learning_rate=0.01)
    numEpochs = 0
    batchSize = 0
    historyOne, historyTwo, historyThree = History(), History(), History()
    totalLoss, totalValLoss = pd.DataFrame(), pd.DataFrame()
    y_hat = []
    responseLabel['text'] = "Reset!"
    print("Reset!")


def WaveletTransform(data, levels, threshold=0.63, wavelet='coif3'):
    reconstructedData = pd.DataFrame()
    for i in range(data.shape[1]):
        threshold = threshold * np.nanmax(data[:, i])
        coefficients = pywt.wavedec(
            data[:, i], wavelet, mode='per', level=levels)
        coefficients[1:] = (pywt.threshold(
            i, value=threshold, mode='soft') for i in coefficients[1:])
        reconstructedColumn = pywt.waverec(coefficients, wavelet, mode='per')
        reconstructedData = pd.concat(
            [reconstructedData, pd.DataFrame(reconstructedColumn)], axis=1)
    return reconstructedData


def coefficient_of_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return (1 - SS_res/(SS_tot+K.epsilon()))

# ! Tkinter Objects


# ? Window Variables
window = tk.Tk()
window.title("Magic Window")

# ? Colour Scheme
backgroundColor = '#000000'
foregroundColor = '#ffffff'
buttonColor = '#808080'

# ? Title
titleFrame = tk.Frame(window, pady=40)
titleFrame.grid(row=0, columnspan=1, sticky='EW')

title = tk.Label(titleFrame, text="Magic Window",
                 font=("Arial", 35, "bold"), width=40)
signature = tk.Label(titleFrame, text="By Jordan Rowley",
                     font=("Helvetica", 15))

title.pack()
signature.pack()

# ? Choose Ticker
tickerFrame = tk.Frame(window, pady=10)
tickerFrame.grid(row=1, columnspan=1, sticky='EW')

tickerLabel = tk.Label(tickerFrame, text="Ticker Symbol:",
                       font=("Helvetica", 20))
tickerEntry = tk.Entry(tickerFrame)

tickerLabel.pack(side="left", padx=(100, 0))
tickerEntry.pack(side="right", padx=(0, 100))

# ? Choose Look Ahead
lookAheadFrame = tk.Frame(window, pady=10)
lookAheadFrame.grid(row=2, columnspan=1, sticky='EW')

lookAheadLabel = tk.Label(lookAheadFrame, text="Days to prediction:",
                          font=("Helvetica", 20))
lookAheadEntry = tk.Entry(lookAheadFrame)

lookAheadLabel.pack(side="left", padx=(100, 0))
lookAheadEntry.pack(side="right", padx=(0, 100))

# ? Choose Exposure Size
exposureSizeFrame = tk.Frame(window, pady=10)
exposureSizeFrame.grid(row=3, columnspan=1, sticky='EW')

exposureSizeLabel = tk.Label(exposureSizeFrame, text="Exposure size:",
                             font=("Helvetica", 20))
exposureSizeEntry = tk.Entry(exposureSizeFrame)

exposureSizeLabel.pack(side="left", padx=(100, 0))
exposureSizeEntry.pack(side="right", padx=(0, 100))

# ? Choose Number of Training Epochs
numEpochsFrame = tk.Frame(window, pady=10)
numEpochsFrame.grid(row=4, columnspan=1, sticky='EW')

numEpochsLabel = tk.Label(numEpochsFrame, text="Number of training epochs:",
                          font=("Helvetica", 20))
numEpochsEntry = tk.Entry(numEpochsFrame)

numEpochsLabel.pack(side="left", padx=(100, 0))
numEpochsEntry.pack(side="right", padx=(0, 100))

# ? Choose Batch Size
batchSizeFrame = tk.Frame(window, pady=10)
batchSizeFrame.grid(row=5, columnspan=1, sticky='EW')

batchSizeLabel = tk.Label(batchSizeFrame, text="Batch size:",
                          font=("Helvetica", 20))
batchSizeEntry = tk.Entry(batchSizeFrame)

batchSizeLabel.pack(side="left", padx=(100, 0))
batchSizeEntry.pack(side="right", padx=(0, 100))

# ? Update Variables
# updateFrame = tk.Frame(window, pady=10)
# updateFrame.grid(row=6, columnspan=1, sticky='EW')

# updateButton = tk.Button(updateFrame, text="Update Variables", font=(
#     "Helvetica", 20), command=UpdateVariables, width=25)

# updateButton.pack(pady=(20, 0))

# ? Run
runFrame = tk.Frame(window, pady=10)
runFrame.grid(row=6, columnspan=1, sticky='EW')

runButton = tk.Button(runFrame, text="Run", font=(
    "Helvetica", 20), command=Run, width=25)

runButton.pack(pady=(30, 0))

# ? Save / Load Model
# saveLoadFrame = tk.Frame(window, pady=10)
# saveLoadFrame.grid(row=8, columnspan=1, sticky='EW')

# saveButton = tk.Button(saveLoadFrame, text="Save Model", font=("Helvetica", 20),
#                        command=SaveModel, width=11)
# loadButton = tk.Button(saveLoadFrame, text="Load Model", font=("Helvetica", 20),
#                        command=LoadModel, width=11)

# saveButton.pack(side="left", padx=(250, 0))
# loadButton.pack(side="right", padx=(0, 250))

# ? Predict and Plot
# predictAndPlotFrame = tk.Frame(window, pady=10)
# predictAndPlotFrame.grid(row=7, columnspan=1, sticky='EW')

# predictAndPlotButton = tk.Button(
#     predictAndPlotFrame, text="Predict and Plot", font=("Helvetica", 20), command=PredictAndPlot, width=25, height=1)

# predictAndPlotButton.pack()

# ? Reset
resetFrame = tk.Frame(window, pady=10)
resetFrame.grid(row=7, columnspan=1, sticky='EW')

resetButton = tk.Button(resetFrame, text="Reset", font=("Helvetica", 20),
                        command=Reset, width=25, height=1)

resetButton.pack()

# ? Dynamic Repsponse Label
responseFrame = tk.Frame(window, pady=10)
responseFrame.grid(row=8, columnspan=1, sticky='EW')

responseLabel = tk.Label(responseFrame, text="Ready", font=("Helvetica", 20))

responseLabel.pack(pady=(20, 30))

# ? Mainloop
window.mainloop()
