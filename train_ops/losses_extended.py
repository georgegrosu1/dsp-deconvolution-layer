from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.losses import MeanSquaredError
from signal_extended import power_spectrum_db


class PowSpectrumMSE(Loss):
    def call(self, y_true, y_pred):
        mse = MeanSquaredError()
        y_true_psd = power_spectrum_db(y_true)
        y_pred_psd = power_spectrum_db(y_pred)
        return mse(y_true_psd, y_pred_psd)
