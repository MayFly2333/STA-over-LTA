import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# constants
nch = 1704
scale = 10
time = np.arange(0,60,0.01)
fs = 100

# basic small funcs
def phase_to_strain(phase, lam=1550, e=0.78, n=1.468, gl=10.0):
    """将相位数据转换为应变"""
    phase_unwrap = np.unwrap(phase, axis=0)
    strain = phase_unwrap * (lam * 1e-9) / (4 * np.pi * n * e * gl)
    return strain

def sta_lta_full(signal_full, fs, N_s_sec=1.0, N_l_sec=10.0, valid_len=None):
    """计算严格因果的 STA/LTA 比值，并只返回目标段长度"""
    N_s = int(N_s_sec * fs)
    N_l = int(N_l_sec * fs)
    abs_sig = np.abs(signal_full)

    cumsum = np.cumsum(np.insert(abs_sig, 0, 0)) # do cumulative sum of the abs input
    STA = (cumsum[N_s:] - cumsum[:-N_s]) / N_s # cumsum[N_s:]
    LTA = (cumsum[N_l:] - cumsum[:-N_l]) / N_l

    diff = len(STA) - len(LTA)
    if diff > 0:
        LTA = np.concatenate([np.full(diff, np.nan), LTA])
    elif diff < 0:
        STA = np.concatenate([np.full(-diff, np.nan), STA])

    STA = np.concatenate([np.full(N_l, np.nan), STA])
    LTA = np.concatenate([np.full(N_l, np.nan), LTA])

    R = STA / (LTA + 1e-8)
    if valid_len is not None:
        R = R[-valid_len:]
    return R

# big funcs
# 普通的 波形未处理+STA/LTA
def plot_sta_lta_window(data_before, data_now, fs, scale=50,
                        channels_to_plot=range(0,1704,100),
                        N_s_sec=1.0, N_l_sec=10.0,
                        sta_offset_frac=0.25):
    """
    与之前功能相同，但在绘 STA/LTA 时加入垂直偏移，避免与原始波形重合。
    sta_offset_frac: 相对于通道间距的偏移比例（默认 0.25，即 25%）
    """
    nch, ns = data_now.shape

    subset_before = np.hstack([np.zeros((nch,1)), np.diff(data_before, axis=1)])
    subset_now = np.hstack([np.zeros((nch,1)), np.diff(data_now, axis=1)])
    full_signal = np.concatenate([subset_before, subset_now], axis=1)

    valid_len = subset_now.shape[1]
    time = np.arange(valid_len) / fs

    # 计算绘图中通道索引的最小间隔，用作偏移参考
    ch_indices = np.array(list(channels_to_plot))
    if ch_indices.size >= 2:
        ch_spacing = np.min(np.diff(ch_indices))
    else:
        ch_spacing = 1.0
    sta_offset = sta_offset_frac * ch_spacing

    plt.figure(figsize=(14, 12))
    for i in channels_to_plot:
        # 注意：i 是 channel 索引或用于叠放的基线值
        # 原始 now 信号（标准化后）
        baseline_wave = subset_now[i] * scale / (np.std(subset_now[i]) + 1e-12) + i
        plt.plot(time, baseline_wave, 'b', linewidth=0.4)

        # STA/LTA（取最后 1 分钟部分），并上移 sta_offset
        sta_lta = sta_lta_full(full_signal[i], fs,
                               N_s_sec=N_s_sec,
                               N_l_sec=N_l_sec,
                               valid_len=valid_len)
        plt.plot(time, sta_lta * 50 + i, 'r', linewidth=1)

    plt.title("STA/LTA")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel Index")
    plt.show()

# 拼起来 三个时间窗
def plot_sta_lta_three_windows(data_before, win1, win2, win3, fs,
                               scale=50, channels_to_plot=range(0,1704,100),
                               N_s_sec=1.0, N_l_sec=10.0):
    """
    拼接三个连续一分钟的时间窗画 STA/LTA
    data_before: 前一分钟数据，用于 STA/LTA 因果计算
    win1, win2, win3: 当前三分钟数据
    fs: 采样率
    scale: 蓝线波形放大倍数
    """
    nch = win1.shape[0]

    # 差分并拼接前一分钟 + 三个窗口 这里可能有问题！！！
    subset_before = np.hstack([np.zeros((nch,1)), np.diff(data_before, axis=1)])
    subset1 = np.hstack([np.zeros((nch,1)), np.diff(win1, axis=1)])
    subset2 = np.hstack([np.zeros((nch,1)), np.diff(win2, axis=1)])
    subset3 = np.hstack([np.zeros((nch,1)), np.diff(win3, axis=1)])

    # 拼接三个窗口
    full_signal = np.concatenate([subset_before, subset1, subset2, subset3], axis=1)

    ns_total = subset1.shape[1] + subset2.shape[1] + subset3.shape[1]
    time = np.arange(ns_total) / fs

    plt.figure(figsize=(20, 15))
    for i in channels_to_plot:
        # 蓝线
        baseline_wave = np.concatenate([subset1[i], subset2[i], subset3[i]]) * scale / (np.std(np.concatenate([subset1[i], subset2[i], subset3[i]]))+1e-12) + i
        plt.plot(time, baseline_wave, 'b', linewidth=0.4)

        # 红线 STA/LTA（固定乘 50），使用前一分钟作为因果
        sta_lta = sta_lta_full(full_signal[i], fs, N_s_sec=N_s_sec, N_l_sec=N_l_sec, valid_len=ns_total)
        plt.plot(time, sta_lta*50 + i, 'r', linewidth=1)

    plt.title("STA/LTA - 3 Windows Continuous")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel Index")
    plt.show()

# 每个时间窗单独处理，+spike_removal，拼接画图
# 这个函数跑着太慢了，建议用下面那个＋其他小函数

def plot_sta_lta_multi_windows(data_before, *windows, fs=100,
                               scale=50, channels_to_plot=range(0,1704,100),
                               N_s_sec=1.0, N_l_sec=10.0):
    """
    data_before: 前一分钟数据
    *windows: 任意数量的后续分钟数据
    """
    nch = data_before.shape[0]

    # 差分：前一分钟
    subset_before = np.hstack([np.zeros((nch,1)), np.diff(data_before, axis=1)])

    # 差分 + 拼接所有窗口
    diff_windows = []
    for win in windows:
        diff_win = np.hstack([np.zeros((nch,1)), np.diff(win, axis=1)])
        diff_windows.append(diff_win)

    # 拼成 full_signal（用于因果 STA/LTA）
    full_signal = np.concatenate([subset_before] + diff_windows, axis=1)
    # 做spike_removal!
    full_signal = spike_removal(full_signal)

    # 拼出绘图信号
    concat_windows = np.concatenate(diff_windows, axis=1)
    ns_total = concat_windows.shape[1]
    time = np.arange(ns_total) / fs

    plt.figure(figsize=(20, 15))
    for i in channels_to_plot:

        # 蓝线
        blue = concat_windows[i]
        blue = blue * scale / (np.std(blue) + 1e-12) + i
        plt.plot(time, blue, 'b', linewidth=0.4)

        # STA/LTA
        sta_lta = sta_lta_full(full_signal[i], fs, 
                               N_s_sec=N_s_sec, N_l_sec=N_l_sec,
                               valid_len=ns_total)
        plt.plot(time, sta_lta * 50 + i, 'r', linewidth=1)

    plt.title("STA/LTA - Multi Windows Continuous")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel Index")
    plt.show()

from daspy.core.section import Section

def concat_sec_objects(data_before_sec, *window_secs, do_spike=True):

    # ----------- 1. 取出 data 并差分 -----------
    def diff_data(data):
        nch = data.shape[0]
        return np.hstack([np.zeros((nch,1)), np.diff(data, axis=1)])

    # 前一分钟
    diff_before = diff_data(data_before_sec.data)

    # windows
    diff_wins = [diff_data(sec.data) for sec in window_secs]

    # ----------- 2. 拼接 -----------
    full_data = np.concatenate([diff_before] + diff_wins, axis=1)

    # ----------- 3. spike removal -----------
    if do_spike:
        full_data = spike_removal(full_data)

    # ----------- 4. 构造新的 Section 对象 -----------
    new_sec = Section(
        data = full_data,
        dx   = 10,   
        fs   = 100
    )

    return new_sec
  
from daspy.core.section import Section
import numpy as np

def stack_section(sec, stack_half_width=1):
    """
    对 Section.data 在 channel 方向做 stacking（平均），
    返回一个新的 Section 对象。

    Parameters
    ----------
    sec : Section
        输入的 Section 对象
    stack_half_width : int
        左右各多少条道参与平均，比如=1就是3道平均

    Returns
    -------
    Section
        经过 stacking 的新的 Section
    """

    data = sec.data
    nch, ns = data.shape

    stack_size = 2 * stack_half_width + 1
    new_channels = nch - 2 * stack_half_width

    if new_channels <= 0:
        raise ValueError("stack_half_width 太大，stack 后没有通道了")

    stacked_data = np.zeros((new_channels, ns))

    # 对每一个目标通道做 stacking
    for ch in range(new_channels):
        low  = ch
        high = ch + stack_size
        stacked_data[ch] = np.mean(data[low:high, :], axis=0)

    # 生成新的 Section
    new_sec = Section(
        data = stacked_data,
        dx   = 10,
        fs = 100
    )

    return new_sec

# stack related
def sta_lta_full_stacked(data, center_ch, stack_half_width, fs,
                         N_s_sec=1.0, N_l_sec=10.0, valid_len=None):
    """
    对中心通道附近 (stack_half_width*2+1) 条通道做平均，
    再用原 sta_lta_full 计算 STA/LTA。
    """
    nch = data.shape[0]

    # 选取 stacking 范围
    low  = max(0, center_ch - stack_half_width)
    high = min(nch, center_ch + stack_half_width + 1)

    # 平均后的信号（严格按照你的逻辑）
    stacked_signal = np.mean(data[low:high, :], axis=0)

    # 调用你原来的 STA/LTA 核心函数
    return sta_lta_full(
        stacked_signal,
        fs,
        N_s_sec=N_s_sec, 
        N_l_sec=N_l_sec,
        valid_len=valid_len
    )

def plot_sta_lta_highlight_stack(
    data_before, data_now, fs, scale=50,
    channels_to_plot=range(0,1704,100),
    stack_half_width=10,            # 新增：左右各多少条道参与平均
    N_s_sec=1.0, N_l_sec=10.0,
    sta_threshold=3.0,
    highlight_color='orange'
):
    nch, ns = data_now.shape

    # 差分
    subset_before = np.hstack([np.zeros((nch,1)), np.diff(data_before, axis=1)])
    subset_now    = np.hstack([np.zeros((nch,1)), np.diff(data_now, axis=1)])
    full_signal   = np.concatenate([subset_before, subset_now], axis=1)

    valid_len = subset_now.shape[1]
    time = np.arange(valid_len) / fs

    plt.figure(figsize=(14, 12))

    for ch in channels_to_plot:
        # ----------- 通道堆叠（平均）-----------
        low  = max(0, ch - stack_half_width)
        high = min(nch, ch + stack_half_width + 1)

        stacked_now  = np.mean(subset_now[low:high, :], axis=0)
        stacked_full = np.mean(full_signal[low:high, :], axis=0)

        # ----------- 蓝线：平均后的波形 -----------
        baseline_wave = (
            stacked_now * scale / (np.std(stacked_now) + 1e-12)
            + ch
        )

        # ----------- STA/LTA -----------
        sta_lta = sta_lta_full_stacked(
            full_signal,
            center_ch=ch,
            stack_half_width=stack_half_width,
            fs=fs,
            N_s_sec=N_s_sec,
            N_l_sec=N_l_sec,
            valid_len=valid_len
        )


        plt.plot(time, sta_lta * 50 + ch, 'r', linewidth=1)

        # ----------- 高亮 STA/LTA 超阈值部分 -----------
        above = sta_lta > sta_threshold
        below = ~above

        plt.plot(time[below], baseline_wave[below], 'b', linewidth=0.4)
        plt.plot(time[above], baseline_wave[above], highlight_color, linewidth=0.7)

    plt.title(f"STA/LTA (stack {stack_half_width*2+1} channels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel Index")
    plt.show()

#不画STA/LTA的
def plot_stack(
    data_before, data_now, fs, scale=50,
    channels_to_plot=range(0,1704,100),
    stack_half_width=10,            # 新增：左右各多少条道参与平均
    N_s_sec=1.0, N_l_sec=10.0,
    sta_threshold=3.0,
    highlight_color='b'
):
    nch, ns = data_now.shape

    # 差分
    subset_before = np.hstack([np.zeros((nch,1)), np.diff(data_before, axis=1)])
    subset_now    = np.hstack([np.zeros((nch,1)), np.diff(data_now, axis=1)])
    full_signal   = np.concatenate([subset_before, subset_now], axis=1)

    valid_len = subset_now.shape[1]
    time = np.arange(valid_len) / fs

    plt.figure(figsize=(14, 12))

    for ch in channels_to_plot:
        # ----------- 通道堆叠（平均）-----------
        low  = max(0, ch - stack_half_width)
        high = min(nch, ch + stack_half_width + 1)

        stacked_now  = np.mean(subset_now[low:high, :], axis=0)
        stacked_full = np.mean(full_signal[low:high, :], axis=0)

        # ----------- 蓝线：平均后的波形 -----------
        baseline_wave = (
            stacked_now * scale / (np.std(stacked_now) + 1e-12)
            + ch
        )

        # ----------- STA/LTA -----------
        sta_lta = sta_lta_full_stacked(
            full_signal,
            center_ch=ch,
            stack_half_width=stack_half_width,
            fs=fs,
            N_s_sec=N_s_sec,
            N_l_sec=N_l_sec,
            valid_len=valid_len
        )


        #plt.plot(time, sta_lta * 50 + ch, 'r', linewidth=1)

        # ----------- 高亮 STA/LTA 超阈值部分 -----------
        above = sta_lta > sta_threshold
        below = ~above

        plt.plot(time[below], baseline_wave[below], 'b', linewidth=0.4)
        plt.plot(time[above], baseline_wave[above], highlight_color, linewidth=0.7)

    plt.title(f"STA/LTA (stack {stack_half_width*2+1} channels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel Index")
    plt.show()

def plot_sta_lta_stack_highlightPeakTime(
    data_before, data_now, fs, scale=50,
    channels_to_plot=range(0,1704,100),
    stack_half_width=10,
    N_s_sec=1.0, N_l_sec=10.0,
    sta_threshold=3.0,
    highlight_color='orange'
):
    nch, ns = data_now.shape

    # ---------------- 差分 ----------------
    subset_before = np.hstack([np.zeros((nch,1)), np.diff(data_before, axis=1)])
    subset_now    = np.hstack([np.zeros((nch,1)), np.diff(data_now, axis=1)])
    full_signal   = np.concatenate([subset_before, subset_now], axis=1)

    valid_len = subset_now.shape[1]
    time = np.arange(valid_len) / fs

    # 统计每个时间点超过阈值的道数
    sta_lta_all = np.zeros((len(channels_to_plot), valid_len))

    for idx, ch in enumerate(channels_to_plot):
        sta_lta_all[idx] = sta_lta_full_stacked(
            full_signal,
            center_ch=ch,
            stack_half_width=stack_half_width,
            fs=fs,
            N_s_sec=N_s_sec,
            N_l_sec=N_l_sec,
            valid_len=valid_len
        )

    # 对每个时间点统计多少通道超过阈值
    exceed_counts = np.sum(sta_lta_all > sta_threshold, axis=0)

    # 找到最多的那个时间点
    peak_idx = np.argmax(exceed_counts)
    peak_time = peak_idx / fs
    print(f"most likely to happen at： {peak_time:.3f} 秒")
    
    plt.figure(figsize=(14, 12))

    for idx, ch in enumerate(channels_to_plot):
        low  = max(0, ch - stack_half_width)
        high = min(nch, ch + stack_half_width + 1)

        stacked_now  = np.mean(subset_now[low:high, :], axis=0)

        # 蓝线：平均后的波形
        baseline_wave = (
            stacked_now * scale / (np.std(stacked_now) + 1e-12)
            + ch
        )

        sta_lta = sta_lta_all[idx]

        # 超阈值区间
        above = sta_lta > sta_threshold
        below = ~above

        plt.plot(time[below], baseline_wave[below], 'b', linewidth=0.4)
        plt.plot(time[above], baseline_wave[above], highlight_color, linewidth=0.7)

        # 红线 STA/LTA
        plt.plot(time, sta_lta * 50 + ch, 'r', linewidth=1)

    # 画出事件最可能的时间点（竖线）
    plt.axvline(peak_time, color='magenta', linestyle='--', linewidth=2,
                label=f"Peak Event Time = {peak_time:.2f}s")

    plt.title(f"STA/LTA (stack {stack_half_width*2+1} channels) + Event Peak Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel Index")
    plt.legend()
    plt.show()

def plot_sta_lta_stack_highlightPeakTime_cleverThreshold(
    data_before, data_now, fs, scale=50,
    channels_to_plot=range(0,1704,100),
    stack_half_width=10,
    N_s_sec=1.0, N_l_sec=10.0,
    threshold_k=3.0,          # k × std
    threshold_tau=1.0,        # <-- 最低阈值 τ
    highlight_color='orange'
):

    nch, ns = data_now.shape

    # ---------------- 差分 ----------------
    subset_before = np.hstack([np.zeros((nch,1)), np.diff(data_before, axis=1)])
    subset_now    = np.hstack([np.zeros((nch,1)), np.diff(data_now, axis=1)])
    full_signal   = np.concatenate([subset_before, subset_now], axis=1)

    valid_len = subset_now.shape[1]
    time = np.arange(valid_len) / fs

    # 保存所有通道的 STA/LTA
    sta_lta_all = np.zeros((len(channels_to_plot), valid_len))
    thresholds  = np.zeros(len(channels_to_plot))

    # ---------------- 计算 STA/LTA + 每道阈值 ----------------
    for idx, ch in enumerate(channels_to_plot):

        sta_lta_seq = sta_lta_full_stacked(
            full_signal,
            center_ch=ch,
            stack_half_width=stack_half_width,
            fs=fs,
            N_s_sec=N_s_sec,
            N_l_sec=N_l_sec,
            valid_len=valid_len
        )

        sta_lta_all[idx] = sta_lta_seq

        # 每条道的 std
        std_val = np.std(sta_lta_seq)

        # ---- NEW：阈值 = max(k·std, τ) ----
        thresholds[idx] = max(threshold_k * std_val, threshold_tau)

    # ---------------- 统计 peak time ----------------
    exceed_matrix = sta_lta_all > thresholds[:, None]
    exceed_counts = np.sum(exceed_matrix, axis=0)

    peak_idx = np.argmax(exceed_counts)
    peak_time = peak_idx / fs
    print(f"Most likely event time: {peak_time:.3f} s")

    # ---------------- 画图 ----------------
    plt.figure(figsize=(14, 12))

    for idx, ch in enumerate(channels_to_plot):
        low  = max(0, ch - stack_half_width)
        high = min(nch, ch + stack_half_width + 1)

        # 平均后的波形
        stacked_now = np.mean(subset_now[low:high, :], axis=0)
        baseline_wave = stacked_now * scale / (np.std(stacked_now) + 1e-12) + ch

        sta_lta = sta_lta_all[idx]
        threshold = thresholds[idx]

        above = sta_lta > threshold
        below = ~above

        plt.plot(time[below], baseline_wave[below], 'b', linewidth=0.4)
        plt.plot(time[above], baseline_wave[above], highlight_color, linewidth=0.7)

        # 红线 STA/LTA
        plt.plot(time, sta_lta * 50 + ch, 'r', linewidth=1)

    plt.axvline(
        peak_time,
        color='magenta',
        linestyle='--',
        linewidth=2,
        label=f"Peak Event Time = {peak_time:.2f}s"
    )

    plt.title(f"STA/LTA (stack {stack_half_width*2+1}) + adaptive threshold max(k·std, τ)")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel Index")
    plt.legend()
    plt.show()

def plot_sta_lta_stack_highlightPeakTime_autoTau(
    data_before, data_now, fs, scale=50,
    channels_to_plot=range(0,1704,100),
    stack_half_width=10,
    N_s_sec=1.0, N_l_sec=10.0,
    k=3.0,                 # std 系数
    alpha=6.0,             # MAD 系数 → 自动 tau
    event_ratio=0.8,       # 超过 80% 才认为是 event
    use_event_ratio=True,  # 开关
    highlight_color='orange'
):

    nch, ns = data_now.shape

    # ---------------- 差分 ----------------
    subset_before = np.hstack([np.zeros((nch,1)), np.diff(data_before, axis=1)])
    subset_now    = np.hstack([np.zeros((nch,1)), np.diff(data_now, axis=1)])
    full_signal   = np.concatenate([subset_before, subset_now], axis=1)

    valid_len = subset_now.shape[1]
    time = np.arange(valid_len) / fs

    # 保存 STA/LTA
    sta_lta_all = np.zeros((len(channels_to_plot), valid_len))
    thresholds  = np.zeros(len(channels_to_plot))

    # ---------------- 计算 STA/LTA + adaptive threshold ----------------
    for idx, ch in enumerate(channels_to_plot):

        sta_lta_seq = sta_lta_full_stacked(
            full_signal,
            center_ch=ch,
            stack_half_width=stack_half_width,
            fs=fs,
            N_s_sec=N_s_sec,
            N_l_sec=N_l_sec,
            valid_len=valid_len
        )

        sta_lta_all[idx] = sta_lta_seq

        # === robust tau using median + MAD ===
        med = np.median(sta_lta_seq)
        mad = np.median(np.abs(sta_lta_seq - med))

        tau = med + alpha * mad
        std_val = np.std(sta_lta_seq)

        # final threshold
        thresholds[idx] = max(k * std_val, tau)

    # ---------------- Detect event ----------------
    exceed_matrix = sta_lta_all > thresholds[:, None]
    exceed_ratio  = np.mean(exceed_matrix, axis=0)

    if use_event_ratio:
        # -------- 使用 80% 规则 --------
        event_mask = exceed_ratio > event_ratio

        if np.any(event_mask):
            peak_idx = np.argmax(exceed_ratio)
            peak_time = peak_idx / fs
            print(f"[80% rule] Event detected at {peak_time:.3f}s (ratio={exceed_ratio[peak_idx]:.3f})")
        else:
            peak_time = None
            print("[80% rule] No event detected.")
    else:
        # -------- 不使用 80%，只取 exceed_ratio 最大点 --------
        peak_idx = np.argmax(exceed_ratio)
        peak_time = peak_idx / fs
        print(f"[NO ratio rule] Peak at {peak_time:.3f}s (ratio={exceed_ratio[peak_idx]:.3f})")

    # ---------------- Plot ----------------
    plt.figure(figsize=(14, 12))

    for idx, ch in enumerate(channels_to_plot):

        low  = max(0, ch - stack_half_width)
        high = min(nch, ch + stack_half_width + 1)

        stacked_now = np.mean(subset_now[low:high, :], axis=0)
        baseline_wave = stacked_now * scale / (np.std(stacked_now)+1e-12) + ch

        sta_lta = sta_lta_all[idx]
        thr = thresholds[idx]

        above = sta_lta > thr
        below = ~above

        plt.plot(time[below], baseline_wave[below], 'b', linewidth=0.4)
        plt.plot(time[above], baseline_wave[above], highlight_color, linewidth=0.7)

        plt.plot(time, sta_lta * 50 + ch, 'r', linewidth=1)

    if peak_time is not None:
        plt.axvline(peak_time, color='magenta', linestyle='--', linewidth=2,
                    label=f"Peak = {peak_time:.2f}s")

    plt.title("STA/LTA + adaptive tau + optional event proportion rule")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel Index")
    plt.legend()
    plt.show()

# example
# S波 

plot_sta_lta_stack_highlightPeakTime(
    data_before = sec_phase11_45_1m_17000m__55188.data,
    data_now    = sec_phase11_46_1m_17000m__55189.data,
    fs = 100,
    scale = 7,
    channels_to_plot = range(0,1704,50),
    stack_half_width = 200,      
    sta_threshold = 1.3
)
