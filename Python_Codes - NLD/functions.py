import torch
from sklearn.preprocessing import StandardScaler
import math
import numpy as np


def map(signal_bit, modu_way):
    '''
    :param signal_bit: the bit signal ,shape = (ofdm_sym_num, data_sub_num*bit_to_sym[modu_way])
    :param modu_way:  0:bpsk, 1:qpsk, 2:16qam, 3:64qam
    :return: output , pilot_symbol
             output = signal_symbol, shape =(ofdm_sym_num, data_sub_num)
    '''

    if modu_way == 0:
        output = map_bpsk(signal_bit)
    elif modu_way == 1:
        output = map_qpsk(signal_bit)
    elif modu_way == 2:
        output = map_16qam(signal_bit)
    elif modu_way == 3:
        output = map_64qam(signal_bit)
    else:
        print('the input of modu_way is error')
        output = 1
    return output


def map_bpsk(signal_bit):
    output = np.empty_like(signal_bit, dtype="complex64")
    for m in range(signal_bit.shape[0]):
        for n in range(signal_bit.shape[1]):
            if signal_bit[m, n] == 0:
                output[m, n] = -1 + 0j
            else:
                output[m, n] = 1 + 0j
    return output


def map_qpsk(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1] / 2)
    x = signal_bit.reshape(c, d, 2)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :]
            if (a == [0, 0]).all():
                output[m, n] = complex(-math.sqrt(2)/2, -math.sqrt(2)/2)
            elif (a == [0, 1]).all():
                output[m, n] = complex(-math.sqrt(2)/2, math.sqrt(2)/2)
            elif (a == [1, 1]).all():
                output[m, n] = complex(math.sqrt(2) / 2, math.sqrt(2) / 2)
            else:
                output[m, n] = complex(math.sqrt(2) / 2, -math.sqrt(2) / 2)
    return output


def map_16qam(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1]/4)
    x = signal_bit.reshape(c, d, 4)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :2]
            if (a == [0, 0]).all():
                real = -3
            elif (a == [0, 1]).all():
                real = -1
            elif (a == [1, 1]).all():
                real = 1
            else:
                real = 3
            b = x[m, n, 2:]
            if (b == [0, 0]).all():
                imag = -3
            elif (b == [0, 1]).all():
                imag = -1
            elif (b == [1, 1]).all():
                imag = 1
            else:
                imag = 3
            output[m, n] = complex(real, imag)/math.sqrt(10)
    return output


def map_64qam(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1]/6)
    x = signal_bit.reshape(c, d, 6)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :3]
            if (a == [0, 0, 0]).all():
                real = -7
            elif (a == [0, 0, 1]).all():
                real = -5
            elif (a == [0, 1, 1]).all():
                real = -3
            elif (a == [0, 1, 0]).all():
                real = -1
            elif (a == [1, 0, 0]).all():
                real = 7
            elif (a == [1, 0, 1]).all():
                real = 5
            elif (a == [1, 1, 1]).all():
                real = 3
            else:
                real = 1
            b = x[m, n, 3:]
            if (b == [0, 0, 0]).all():
                imag = -7
            elif (b == [0, 0, 1]).all():
                imag = -5
            elif (b == [0, 1, 1]).all():
                imag = -3
            elif (b == [0, 1, 0]).all():
                imag = -1
            elif (b == [1, 0, 0]).all():
                imag = 7
            elif (b == [1, 0, 1]).all():
                imag = 5
            elif (b == [1, 1, 1]).all():
                imag = 3
            else:
                imag = 1
            output[m, n] = complex(real, imag)/math.sqrt(84)
    return output


def demap(signal_symbol, modu_way):
    '''
    :param signal_symbol: the symbol signal ,shape = (ofdm_sym_num, data_sub_num)
    :param modu_way:  0:bpsk, 1:qpsk, 2:16qam, 3:64qam
    :return: output
             output = signal_bit, shape =(ofdm_sym_num, data_sub_num*bit_to_sym[modu_way])
    '''
    if signal_symbol.ndim == 1:
        signal_symbol = signal_symbol[np.newaxis, :]
    if modu_way == 0:
        output = demap_bpsk(signal_symbol)
    elif modu_way == 1:
        output = demap_qpsk(signal_symbol)
    elif modu_way == 2:
        output = demap_16qam(signal_symbol)
    elif modu_way == 3:
        output = demap_64qam(signal_symbol)
    else:
        print('the input of modu_way is error')
        output = 1
    return output


def demap_bpsk(x):
    output = np.empty_like(x, dtype="int")
    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            if x[m, n].real >= 0:
                output[m, n] = 1
            else:
                output[m, n] = 0
    return output


def demap_qpsk(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 2), dtype="int")
    for m in range(c):
        for n in range(d):
            a = x[m, n].real
            b = x[m, n].imag
            if (a <= 0) & (b <= 0):
                output[m, n, :] = [0, 0]
            elif (a <= 0) & (b > 0):
                output[m, n, :] = [0, 1]
            elif (a > 0) & (b > 0):
                output[m, n, :] = [1, 1]
            else:
                output[m, n, :] = [1, 0]
    output = output.reshape(c, int(2*d))
    return output


def demap_16qam(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 4), dtype="int")
    for m in range(c):
        for n in range(d):
            a = math.sqrt(10)*x[m, n].real
            if a <= -2:
                output[m, n, :2] = [0, 0]
            elif (a <= 0) & (a > -2):
                output[m, n, :2] = [0, 1]
            elif (a <= 2) & (a > 0):
                output[m, n, :2] = [1, 1]
            else:
                output[m, n, :2] = [1, 0]
            b = math.sqrt(10)*x[m, n].imag
            if b <= -2:
                output[m, n, 2:] = [0, 0]
            elif (b <= 0) & (b > -2):
                output[m, n, 2:] = [0, 1]
            elif (b <= 2) & (b > 0):
                output[m, n, 2:] = [1, 1]
            else:
                output[m, n, 2:] = [1, 0]
    output = output.reshape((c, int(4*d)))
    return output


def demap_64qam(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 6), dtype="int")
    for m in range(c):
        for n in range(d):
            a = math.sqrt(84)*x[m, n].real
            if a <= -6:
                output[m, n, :3] = [0, 0, 0]
            elif (a > -6) & (a <= -4):
                output[m, n, :3] = [0, 0, 1]
            elif (a > -4) & (a <= -2):
                output[m, n, :3] = [0, 1, 1]
            elif (a > -2) & (a <= 0):
                output[m, n, :3] = [0, 1, 0]
            elif (a > 0) & (a <= 2):
                output[m, n, :3] = [1, 1, 0]
            elif (a > 2) & (a <= 4):
                output[m, n, :3] = [1, 1, 1]
            elif (a > 4) & (a <= 6):
                output[m, n, :3] = [1, 0, 1]
            else:
                output[m, n, :3] = [1, 0, 0]
            b = math.sqrt(84) * x[m, n].imag
            if b <= -6:
                output[m, n, 3:] = [0, 0, 0]
            elif (b > -6) & (b <= -4):
                output[m, n, 3:] = [0, 0, 1]
            elif (b > -4) & (b <= -2):
                output[m, n, 3:] = [0, 1, 1]
            elif (b > -2) & (b <= 0):
                output[m, n, 3:] = [0, 1, 0]
            elif (b > 0) & (b <= 2):
                output[m, n, 3:] = [1, 1, 0]
            elif (b > 2) & (b <= 4):
                output[m, n, 3:] = [1, 1, 1]
            elif (b > 4) & (b <= 6):
                output[m, n, 3:] = [1, 0, 1]
            else:
                output[m, n, 3:] = [1, 0, 0]
    output = output.reshape(c, int(6*d))
    return output


