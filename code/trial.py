# def exponential_smoothing(data, alpha):
#     smoothed = [data[0]]  # Initial forecast is set to the first data point
#     for i in range(1, len(data)):
#         smoothed_value = alpha * data[i] + (1 - alpha) * smoothed[-1]
#         smoothed.append(smoothed_value)
#     return smoothed

# # Example light curve data
# light_curve = [10, 10.5, 11, 12, 11, 10.5, 10]  # Example with a flare at the fourth point
# alpha = 0.3

# smoothed_light_curve = exponential_smoothing(light_curve, alpha)

# residuals = [abs(actual - smoothed) for actual, smoothed in zip(light_curve, smoothed_light_curve)]

# import numpy as np

# # Example threshold based on standard deviation
# threshold = np.mean(residuals) + 2 * np.std(residuals)

# flares = [i for i, residual in enumerate(residuals) if residual > threshold]


# import matplotlib.pyplot as plt

# plt.plot(light_curve, label='Original Light Curve')
# plt.plot(smoothed_light_curve, label='Smoothed Light Curve', linestyle='--')
# for flare in flares:
#     plt.axvline(x=flare, color='r', linestyle=':', label='Detected Flare' if flare == flares[0] else "")
# plt.xlabel('Time')
# plt.ylabel('Brightness')
# plt.title('Flare Detection in Light Curve')
# plt.legend()
# plt.show()

# import os, glob
# print(os.path.exists(r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\forced_lc'))

# lc_path = input("Enter path for unprocessed lightcurves... ")
# print(os.path.exists(lc_path))
# while not glob.glob(f'{lc_path}/*.dat') or not os.path.exists(lc_path):
#     lc_path = input("No files found. Enter path for unprocessed lightcurves... ")



# from concurrent.futures import ProcessPoolExecutor
# import time

# def compute_square(n):
#     return n * n

# if __name__ == '__main__':
#     t0=time.time()
#     with ProcessPoolExecutor(max_workers=16) as executor:
#         futures = [executor.submit(compute_square, i) for i in range(1000)]
#         for future in futures:
#             print(future.result())
#     print(time.time()-t0)

#     t1=time.time()
#     for i in range(1000):
#         print(compute_square(i))
#     print(time.time()-t1)

import os

print(os.path.exists(r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\forced_lc_figs\100330391706647456.png'))