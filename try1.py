import tkinter as tk
from tkinter import ttk, messagebox
import random
from typing import List, Tuple

# ===================== 从try1(1).py移植的S-AES核心算法 =====================
# 兼容性开关
ENDIAN = "big"  # "big" | "little"   # 解析0xABCD的端序
LAYOUT = "colmajor"  # "colmajor" | "rowmajor"  # 2x2状态装填方式

# S-Box与逆S-Box
S_BOX = [0x9, 0x4, 0xA, 0xB, 0xD, 0x1, 0x8, 0x5, 0x6, 0x2, 0x0, 0x3, 0xC, 0xE, 0xF, 0x7]
INV_S_BOX = [0xA, 0x5, 0x9, 0xB, 0x1, 0x7, 0x8, 0xF, 0x6, 0x0, 0x2, 0x3, 0xC, 0x4, 0xD, 0xE]

# GF(2^4)乘法的模多项式: x^4 + x + 1 => 0x13
GF_MOD = 0x13


# 工具：端序/布局映射
def _word16_to_nibbles(x: int) -> List[int]:
    """按端序把0xABCD拆成A,B,C,D四个nibble"""
    x &= 0xFFFF
    if ENDIAN == "big":
        A = (x >> 12) & 0xF
        B = (x >> 8) & 0xF
        C = (x >> 4) & 0xF
        D = x & 0xF
    else:  # little-endian解释
        C = (x >> 12) & 0xF
        D = (x >> 8) & 0xF
        A = (x >> 4) & 0xF
        B = x & 0xF
    return [A, B, C, D]


def _nibbles_to_word16(ns: List[int]) -> int:
    A, B, C, D = [n & 0xF for n in ns]
    if ENDIAN == "big":
        return (A << 12) | (B << 8) | (C << 4) | D
    else:
        return (C << 12) | (D << 8) | (A << 4) | B & 0xF


def _pack_state(ns: List[int]) -> List[List[int]]:
    """按布局把四个nibble装入2x2状态矩阵"""
    A, B, C, D = ns
    if LAYOUT == "colmajor":
        return [[A, C], [B, D]]  # 列主序
    else:
        return [[A, B], [C, D]]  # 行主序


def _unpack_state(S: List[List[int]]) -> List[int]:
    if LAYOUT == "colmajor":
        A, C = S[0]
        B, D = S[1]
    else:
        A, B = S[0]
        C, D = S[1]
    return [A, B, C, D]


def _map_in(x16: int) -> int:
    """外部16bit→内部顺序（用于入口）"""
    return _nibbles_to_word16(_unpack_state(_pack_state(_word16_to_nibbles(x16))))


def _map_out(x16: int) -> int:
    """内部顺序→外部16bit（用于出口）"""
    return _nibbles_to_word16(_unpack_state(_pack_state(_word16_to_nibbles(x16))))


# GF(2^4)基本运算
def gmul(a: int, b: int) -> int:
    """GF(2^4)上乘法，模多项式x^4 + x + 1 (0x13)"""
    a &= 0xF;
    b &= 0xF
    res = 0
    for _ in range(4):
        if (b & 1):
            res ^= a
        b >>= 1
        carry = (a & 0x8)  # 最高位是否溢出
        a = (a << 1) & 0xF
        if carry:
            a ^= (GF_MOD & 0xF)  # 只保留低4位
    return res & 0xF


# 核心变换
def sub_nibbles(state: int) -> int:
    """对16bit的4个nibble分别做S-Box"""
    ns = _word16_to_nibbles(state)
    ns = [S_BOX[n] for n in ns]
    return _nibbles_to_word16(ns)


def inv_sub_nibbles(state: int) -> int:
    ns = _word16_to_nibbles(state)
    ns = [INV_S_BOX[n] for n in ns]
    return _nibbles_to_word16(ns)


def shift_rows(state: int) -> int:
    """2x2时第二行循环左移1：n0 n1 n2 n3 -> n0 n3 n2 n1"""
    n0, n1, n2, n3 = _word16_to_nibbles(state)
    ns = [n0, n3, n2, n1]
    return _nibbles_to_word16(ns)


# 对于2x2，inv_shift_rows与shift_rows一致
inv_shift_rows = shift_rows


def mix_columns(state: int) -> int:
    """按列混淆，矩阵[[1,4],[4,1]]"""
    ns = _word16_to_nibbles(state)
    S = _pack_state(ns)  # 2x2矩阵
    # 第一列
    x, y = S[0][0] & 0xF, S[1][0] & 0xF
    S[0][0] = (gmul(1, x) ^ gmul(4, y)) & 0xF
    S[1][0] = (gmul(4, x) ^ gmul(1, y)) & 0xF
    # 第二列
    x, y = S[0][1] & 0xF, S[1][1] & 0xF
    S[0][1] = (gmul(1, x) ^ gmul(4, y)) & 0xF
    S[1][1] = (gmul(4, x) ^ gmul(1, y)) & 0xF
    return _nibbles_to_word16(_unpack_state(S))


def inv_mix_columns(state: int) -> int:
    """按列逆混淆，矩阵[[9,2],[2,9]]"""
    ns = _word16_to_nibbles(state)
    S = _pack_state(ns)
    # 第一列
    x, y = S[0][0] & 0xF, S[1][0] & 0xF
    S[0][0] = (gmul(9, x) ^ gmul(2, y)) & 0xF
    S[1][0] = (gmul(2, x) ^ gmul(9, y)) & 0xF
    # 第二列
    x, y = S[0][1] & 0xF, S[1][1] & 0xF
    S[0][1] = (gmul(9, x) ^ gmul(2, y)) & 0xF
    S[1][1] = (gmul(2, x) ^ gmul(9, y)) & 0xF
    return _nibbles_to_word16(_unpack_state(S))


def add_round_key(state: int, k: int) -> int:
    return (state ^ k) & 0xFFFF


# 密钥扩展
RCON1, RCON2 = 0x80, 0x30  # Rcon：对8-bit字


def rot_nib(b: int) -> int:
    """8-bit：高低nibble交换"""
    b &= 0xFF
    return ((b & 0xF) << 4) | ((b >> 4) & 0xF)


def sub_nib(b: int) -> int:
    """8-bit：对两半nibble逐个过S-Box"""
    hi = (b >> 4) & 0xF
    lo = b & 0xF
    return ((S_BOX[hi] & 0xF) << 4) | (S_BOX[lo] & 0xF)


def key_expansion(key16: int) -> Tuple[int, int, int]:
    """输入16-bit主密钥→3个16-bit轮密钥K0,K1,K2"""
    key16 &= 0xFFFF
    w0 = (key16 >> 8) & 0xFF
    w1 = key16 & 0xFF
    w2 = w0 ^ sub_nib(rot_nib(w1)) ^ RCON1
    w3 = w2 ^ w1
    w4 = w2 ^ sub_nib(rot_nib(w3)) ^ RCON2
    w5 = w4 ^ w3
    K0 = ((w0 << 8) | w1) & 0xFFFF
    K1 = ((w2 << 8) | w3) & 0xFFFF
    K2 = ((w4 << 8) | w5) & 0xFFFF
    return K0, K1, K2


# 单块加/解密
def s_aes_encrypt(plaintext: int, key: int) -> int:
    """S-AES加密：2轮结构（首轮含Mix，末轮无Mix）"""
    plaintext = _map_in(plaintext)
    k0, k1, k2 = key_expansion(key)
    state = add_round_key(plaintext, k0)

    # 第1轮
    state = sub_nibbles(state)
    state = shift_rows(state)
    state = mix_columns(state)
    state = add_round_key(state, k1)

    # 第2轮（无Mix）
    state = sub_nibbles(state)
    state = shift_rows(state)
    state = add_round_key(state, k2)

    return _map_out(state)


def s_aes_decrypt(ciphertext: int, key: int) -> int:
    ciphertext = _map_in(ciphertext)
    k0, k1, k2 = key_expansion(key)

    # 逆第2轮
    state = add_round_key(ciphertext, k2)
    state = inv_shift_rows(state)
    state = inv_sub_nibbles(state)

    # 逆第1轮
    state = add_round_key(state, k1)
    state = inv_mix_columns(state)
    state = inv_shift_rows(state)
    state = inv_sub_nibbles(state)

    # 最终轮密钥加
    state = add_round_key(state, k0)
    return _map_out(state)


# ===================== 原有功能适配 =====================
# 字符串处理（使用PKCS#7填充）
def str_to_blocks(s: str) -> List[int]:
    """ASCII字符串→16-bit块（PKCS#7，块长2字节）"""
    data = s.encode("ascii")
    pad = 2 - (len(data) % 2)
    if pad == 0:
        pad = 2
    data += bytes([pad]) * pad
    blocks = []
    for i in range(0, len(data), 2):
        blocks.append(((data[i] << 8) | data[i + 1]) & 0xFFFF)
    return blocks


def blocks_to_str(blocks: List[int]) -> str:
    """16-bit块→ASCII字符串（去除PKCS#7填充）"""
    bs = bytearray()
    for b in blocks:
        bs.append((b >> 8) & 0xFF)
        bs.append(b & 0xFF)
    if not bs:
        return ""
    pad = bs[-1]
    if pad in (1, 2) and all(x == pad for x in bs[-pad:]):
        return bs[:-pad].decode("ascii", errors="ignore")
    return bs.decode("ascii", errors="ignore")


def encrypt_str(s: str, key: int) -> str:
    """加密ASCII字符串（输出十六进制字符串）"""
    blocks = str_to_blocks(s)
    cipher_blocks = [s_aes_encrypt(b, key) for b in blocks]
    return ','.join([f"{b:04x}" for b in cipher_blocks])


def decrypt_str(cipher_hex_str: str, key: int) -> str:
    """解密十六进制字符串（还原原始ASCII）"""
    if not cipher_hex_str:
        return ""
    cipher_blocks = [int(b, 16) for b in cipher_hex_str.split(',')]
    plain_blocks = [s_aes_decrypt(b, key) for b in cipher_blocks]
    return blocks_to_str(plain_blocks)


# 多重加密
def double_encrypt(plaintext: int, key32: int) -> int:
    k1 = (key32 >> 16) & 0xffff
    k2 = key32 & 0xffff
    return s_aes_encrypt(s_aes_encrypt(plaintext, k1), k2)


def double_decrypt(ciphertext: int, key32: int) -> int:
    k1 = (key32 >> 16) & 0xffff
    k2 = key32 & 0xffff
    return s_aes_decrypt(s_aes_decrypt(ciphertext, k2), k1)


def triple_encrypt(plaintext: int, key48: int) -> int:
    k1 = (key48 >> 32) & 0xffff
    k2 = (key48 >> 16) & 0xffff
    k3 = key48 & 0xffff
    return s_aes_encrypt(s_aes_decrypt(s_aes_encrypt(plaintext, k1), k2), k3)


def triple_decrypt(ciphertext: int, key48: int) -> int:
    k1 = (key48 >> 32) & 0xffff
    k2 = (key48 >> 16) & 0xffff
    k3 = key48 & 0xffff
    return s_aes_decrypt(s_aes_encrypt(s_aes_decrypt(ciphertext, k3), k2), k1)


# 中间相遇攻击
def meet_in_middle(plaintext: int, ciphertext: int) -> List[int]:
    forward = {}
    for k1 in range(0x10000):
        mid = s_aes_encrypt(plaintext, k1)
        forward.setdefault(mid, []).append(k1)

    possible_keys = []
    for k2 in range(0x10000):
        mid = s_aes_decrypt(ciphertext, k2)
        if mid in forward:
            for k1 in forward[mid]:
                possible_keys.append((k1 << 16) | k2)
    return possible_keys


# CBC模式
def cbc_encrypt(plain_str: str, key: int, iv: int = None) -> Tuple[str, int]:
    """CBC加密（返回: (密文十六进制字符串, IV)）"""
    iv = iv if iv is not None else random.getrandbits(16)
    blocks = str_to_blocks(plain_str)
    cipher_blocks = []
    prev = iv & 0xFFFF
    for b in blocks:
        cipher = s_aes_encrypt(b ^ prev, key)
        cipher_blocks.append(cipher)
        prev = cipher
    cipher_hex = ','.join([f"{b:04x}" for b in cipher_blocks])
    return cipher_hex, iv


def cbc_decrypt(cipher_hex: str, key: int, iv: int) -> str:
    """CBC解密"""
    cipher_blocks = [int(b, 16) for b in cipher_hex.split(',')]
    prev = iv & 0xFFFF
    plain_blocks = []
    for c in cipher_blocks:
        x = s_aes_decrypt(c, key)
        plain_blocks.append(x ^ prev)
        prev = c
    return blocks_to_str(plain_blocks)


def cbc_tamper_test(cipher_hex: str, key: int, iv: int, block_idx=0, new_val=0) -> str:
    cipher_blocks = [int(b, 16) for b in cipher_hex.split(',')]
    if block_idx < 0 or block_idx >= len(cipher_blocks):
        return "无效的块索引"
    tampered = cipher_blocks.copy()
    tampered[block_idx] = new_val
    tampered_hex = ','.join([f"{b:04x}" for b in tampered])
    original = cbc_decrypt(cipher_hex, key, iv)
    tampered_decrypt = cbc_decrypt(tampered_hex, key, iv)
    return f"原始解密: {original}\n篡改后解密: {tampered_decrypt}"


# ===================== GUI界面 =====================
class S_AES_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("S-AES加密工具")
        self.root.geometry("600x550")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_basic = ttk.Frame(self.notebook)
        self.tab_str = ttk.Frame(self.notebook)
        self.tab_double = ttk.Frame(self.notebook)
        self.tab_triple = ttk.Frame(self.notebook)
        self.tab_cbc = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_basic, text="基础加解密")
        self.notebook.add(self.tab_str, text="字符串处理")
        self.notebook.add(self.tab_double, text="双重加密")
        self.notebook.add(self.tab_triple, text="三重加密")
        self.notebook.add(self.tab_cbc, text="CBC模式")

        self.setup_basic_tab()
        self.setup_str_tab()
        self.setup_double_tab()
        self.setup_triple_tab()
        self.setup_cbc_tab()

    # 基础加解密标签页
    def setup_basic_tab(self):
        ttk.Label(self.tab_basic, text="16bit数据 (4位十六进制):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.basic_data = ttk.Entry(self.tab_basic, width=20)
        self.basic_data.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.tab_basic, text="16bit密钥 (4位十六进制):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.basic_key = ttk.Entry(self.tab_basic, width=20)
        self.basic_key.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(self.tab_basic, text="加密", command=self.basic_encrypt).grid(row=2, column=0, padx=5, pady=10)
        ttk.Button(self.tab_basic, text="解密", command=self.basic_decrypt).grid(row=2, column=1, padx=5, pady=10)

        ttk.Label(self.tab_basic, text="结果:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.basic_result = ttk.Entry(self.tab_basic, width=20, state="readonly")
        self.basic_result.grid(row=3, column=1, padx=5, pady=5)

    def basic_encrypt(self):
        try:
            data = int(self.basic_data.get(), 16)
            key = int(self.basic_key.get(), 16)
            if not (0 <= data <= 0xffff and 0 <= key <= 0xffff):
                raise ValueError
            cipher = s_aes_encrypt(data, key)
            self.basic_result.config(state="normal")
            self.basic_result.delete(0, tk.END)
            self.basic_result.insert(0, f"{cipher:04x}")
            self.basic_result.config(state="readonly")
        except:
            messagebox.showerror("错误", "请输入4位十六进制数（0000-FFFF）")

    def basic_decrypt(self):
        try:
            data = int(self.basic_data.get(), 16)
            key = int(self.basic_key.get(), 16)
            if not (0 <= data <= 0xffff and 0 <= key <= 0xffff):
                raise ValueError
            plain = s_aes_decrypt(data, key)
            self.basic_result.config(state="normal")
            self.basic_result.delete(0, tk.END)
            self.basic_result.insert(0, f"{plain:04x}")
            self.basic_result.config(state="readonly")
        except:
            messagebox.showerror("错误", "请输入4位十六进制数（0000-FFFF）")

    # 字符串处理标签页
    def setup_str_tab(self):
        ttk.Label(self.tab_str, text="输入ASCII字符串:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.NW)
        self.str_input = tk.Text(self.tab_str, width=40, height=5)
        self.str_input.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.tab_str, text="16bit密钥 (4位十六进制):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.str_key = ttk.Entry(self.tab_str, width=20)
        self.str_key.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(self.tab_str, text="加密字符串", command=self.str_encrypt).grid(row=2, column=0, padx=5, pady=10)
        ttk.Button(self.tab_str, text="解密字符串", command=self.str_decrypt).grid(row=2, column=1, padx=5, pady=10)

        ttk.Label(self.tab_str, text="加密结果（十六进制）:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.NW)
        self.str_cipher = tk.Text(self.tab_str, width=40, height=3, state="disabled")
        self.str_cipher.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(self.tab_str, text="解密结果:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.NW)
        self.str_result = tk.Text(self.tab_str, width=40, height=3, state="disabled")
        self.str_result.grid(row=4, column=1, padx=5, pady=5)

    def str_encrypt(self):
        try:
            s = self.str_input.get("1.0", tk.END).strip()
            s.encode('ascii')  # 验证ASCII
            key = int(self.str_key.get(), 16)
            if not (0 <= key <= 0xffff):
                raise ValueError
            cipher_hex = encrypt_str(s, key)
            self.str_cipher.config(state="normal")
            self.str_cipher.delete("1.0", tk.END)
            self.str_cipher.insert("1.0", cipher_hex)
            self.str_cipher.config(state="disabled")
        except UnicodeEncodeError:
            messagebox.showerror("错误", "仅支持纯ASCII字符串（不含中文/特殊符号）")
        except:
            messagebox.showerror("错误", "请输入有效的16bit密钥（4位十六进制）")

    def str_decrypt(self):
        try:
            cipher_hex = self.str_cipher.get("1.0", tk.END).strip()
            key = int(self.str_key.get(), 16)
            if not (0 <= key <= 0xffff):
                raise ValueError
            plain_str = decrypt_str(cipher_hex, key)
            self.str_result.config(state="normal")
            self.str_result.delete("1.0", tk.END)
            self.str_result.insert("1.0", plain_str)
            self.str_result.config(state="disabled")
        except:
            messagebox.showerror("错误", "请输入有效的16bit密钥和加密结果")

    # 双重加密标签页
    def setup_double_tab(self):
        ttk.Label(self.tab_double, text="16bit数据 (4位十六进制):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.double_data = ttk.Entry(self.tab_double, width=20)
        self.double_data.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.tab_double, text="32bit密钥 (8位十六进制):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.double_key = ttk.Entry(self.tab_double, width=20)
        self.double_key.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(self.tab_double, text="双重加密", command=self.double_encrypt_btn).grid(row=2, column=0, padx=5,
                                                                                           pady=10)
        ttk.Button(self.tab_double, text="双重解密", command=self.double_decrypt_btn).grid(row=2, column=1, padx=5,
                                                                                           pady=10)

        ttk.Label(self.tab_double, text="结果:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.double_result = ttk.Entry(self.tab_double, width=20, state="readonly")
        self.double_result.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(self.tab_double, text="中间相遇攻击测试:").grid(row=4, column=0, columnspan=2, padx=5, pady=10,
                                                                  sticky=tk.W)
        ttk.Label(self.tab_double, text="明文 (4位十六进制):").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.mim_plain = ttk.Entry(self.tab_double, width=20)
        self.mim_plain.grid(row=5, column=1, padx=5, pady=5)

        ttk.Label(self.tab_double, text="密文 (4位十六进制):").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.mim_cipher = ttk.Entry(self.tab_double, width=20)
        self.mim_cipher.grid(row=6, column=1, padx=5, pady=5)

        ttk.Button(self.tab_double, text="执行攻击", command=self.run_mim).grid(row=7, column=0, columnspan=2, padx=5,
                                                                                pady=10)

        ttk.Label(self.tab_double, text="可能的密钥:").grid(row=8, column=0, padx=5, pady=5, sticky=tk.NW)
        self.mim_result = tk.Text(self.tab_double, width=30, height=3, state="disabled")
        self.mim_result.grid(row=8, column=1, padx=5, pady=5)

    def double_encrypt_btn(self):
        try:
            data = int(self.double_data.get(), 16)
            key32 = int(self.double_key.get(), 16)
            if not (0 <= data <= 0xffff and 0 <= key32 <= 0xffffffff):
                raise ValueError
            cipher = double_encrypt(data, key32)
            self.double_result.config(state="normal")
            self.double_result.delete(0, tk.END)
            self.double_result.insert(0, f"{cipher:04x}")
            self.double_result.config(state="readonly")
        except:
            messagebox.showerror("错误", "数据需为4位十六进制，密钥需为8位十六进制")

    def double_decrypt_btn(self):
        try:
            data = int(self.double_data.get(), 16)
            key32 = int(self.double_key.get(), 16)
            if not (0 <= data <= 0xffff and 0 <= key32 <= 0xffffffff):
                raise ValueError
            plain = double_decrypt(data, key32)
            self.double_result.config(state="normal")
            self.double_result.delete(0, tk.END)
            self.double_result.insert(0, f"{plain:04x}")
            self.double_result.config(state="readonly")
        except:
            messagebox.showerror("错误", "数据需为4位十六进制，密钥需为8位十六进制")

    def run_mim(self):
        try:
            plain = int(self.mim_plain.get(), 16)
            cipher = int(self.mim_cipher.get(), 16)
            if not (0 <= plain <= 0xffff and 0 <= cipher <= 0xffff):
                raise ValueError
            keys = meet_in_middle(plain, cipher)
            self.mim_result.config(state="normal")
            self.mim_result.delete("1.0", tk.END)
            self.mim_result.insert("1.0", "\n".join([f"{k:08x}" for k in keys[:5]]))
            self.mim_result.config(state="disabled")
        except:
            messagebox.showerror("错误", "请输入有效的4位十六进制明密文对")

    # 三重加密标签页
    def setup_triple_tab(self):
        ttk.Label(self.tab_triple, text="16bit数据 (4位十六进制):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.triple_data = ttk.Entry(self.tab_triple, width=20)
        self.triple_data.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.tab_triple, text="48bit密钥 (12位十六进制):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.triple_key = ttk.Entry(self.tab_triple, width=20)
        self.triple_key.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(self.tab_triple, text="三重加密", command=self.triple_encrypt_btn).grid(row=2, column=0, padx=5,
                                                                                           pady=10)
        ttk.Button(self.tab_triple, text="三重解密", command=self.triple_decrypt_btn).grid(row=2, column=1, padx=5,
                                                                                           pady=10)

        ttk.Label(self.tab_triple, text="结果:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.triple_result = ttk.Entry(self.tab_triple, width=20, state="readonly")
        self.triple_result.grid(row=3, column=1, padx=5, pady=5)

    def triple_encrypt_btn(self):
        try:
            data = int(self.triple_data.get(), 16)
            key48 = int(self.triple_key.get(), 16)
            if not (0 <= data <= 0xffff and 0 <= key48 <= 0xffffffffffff):
                raise ValueError
            cipher = triple_encrypt(data, key48)
            self.triple_result.config(state="normal")
            self.triple_result.delete(0, tk.END)
            self.triple_result.insert(0, f"{cipher:04x}")
            self.triple_result.config(state="readonly")
        except:
            messagebox.showerror("错误", "数据需为4位十六进制，密钥需为12位十六进制")

    def triple_decrypt_btn(self):
        try:
            data = int(self.triple_data.get(), 16)
            key48 = int(self.triple_key.get(), 16)
            if not (0 <= data <= 0xffff and 0 <= key48 <= 0xffffffffffff):
                raise ValueError
            plain = triple_decrypt(data, key48)
            self.triple_result.config(state="normal")
            self.triple_result.delete(0, tk.END)
            self.triple_result.insert(0, f"{plain:04x}")
            self.triple_result.config(state="readonly")
        except:
            messagebox.showerror("错误", "数据需为4位十六进制，密钥需为12位十六进制")

    # CBC模式标签页
    def setup_cbc_tab(self):
        ttk.Label(self.tab_cbc, text="输入ASCII字符串:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.NW)
        self.cbc_input = tk.Text(self.tab_cbc, width=40, height=3)
        self.cbc_input.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.tab_cbc, text="16bit密钥 (4位十六进制):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.cbc_key = ttk.Entry(self.tab_cbc, width=20)
        self.cbc_key.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.tab_cbc, text="16bit IV (4位十六进制，留空自动生成):").grid(row=2, column=0, padx=5, pady=5,
                                                                                  sticky=tk.W)
        self.cbc_iv = ttk.Entry(self.tab_cbc, width=20)
        self.cbc_iv.grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(self.tab_cbc, text="CBC加密", command=self.cbc_encrypt_btn).grid(row=3, column=0, padx=5, pady=10)
        ttk.Button(self.tab_cbc, text="CBC解密", command=self.cbc_decrypt_btn).grid(row=3, column=1, padx=5, pady=10)

        ttk.Label(self.tab_cbc, text="密文（十六进制）:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.NW)
        self.cbc_cipher = tk.Text(self.tab_cbc, width=40, height=3, state="disabled")
        self.cbc_cipher.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(self.tab_cbc, text="解密结果:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.NW)
        self.cbc_result = tk.Text(self.tab_cbc, width=40, height=3, state="disabled")
        self.cbc_result.grid(row=5, column=1, padx=5, pady=5)

        ttk.Label(self.tab_cbc, text="篡改测试 (块索引):").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.tamper_idx = ttk.Entry(self.tab_cbc, width=5)
        self.tamper_idx.insert(0, "0")
        self.tamper_idx.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Button(self.tab_cbc, text="测试篡改影响", command=self.test_tamper).grid(row=7, column=0, columnspan=2,
                                                                                     padx=5, pady=10)

        ttk.Label(self.tab_cbc, text="篡改测试结果:").grid(row=8, column=0, padx=5, pady=5, sticky=tk.NW)
        self.tamper_result = tk.Text(self.tab_cbc, width=40, height=3, state="disabled")
        self.tamper_result.grid(row=8, column=1, padx=5, pady=5)

        self.current_cipher_hex = ""
        self.current_iv = 0

    def cbc_encrypt_btn(self):
        try:
            s = self.cbc_input.get("1.0", tk.END).strip()
            s.encode('ascii')  # 验证ASCII
            key = int(self.cbc_key.get(), 16)
            iv_str = self.cbc_iv.get().strip()
            iv = int(iv_str, 16) if iv_str else None

            if not (0 <= key <= 0xffff):
                raise ValueError
            cipher_hex, iv = cbc_encrypt(s, key, iv)
            self.current_cipher_hex = cipher_hex
            self.current_iv = iv

            self.cbc_iv.delete(0, tk.END)
            self.cbc_iv.insert(0, f"{iv:04x}")

            self.cbc_cipher.config(state="normal")
            self.cbc_cipher.delete("1.0", tk.END)
            self.cbc_cipher.insert("1.0", cipher_hex)
            self.cbc_cipher.config(state="disabled")
        except UnicodeEncodeError:
            messagebox.showerror("错误", "仅支持纯ASCII字符串（不含中文/特殊符号）")
        except:
            messagebox.showerror("错误", "请输入有效的密钥和IV（4位十六进制）")

    def cbc_decrypt_btn(self):
        try:
            cipher_hex = self.cbc_cipher.get("1.0", tk.END).strip()
            key = int(self.cbc_key.get(), 16)
            iv = int(self.cbc_iv.get(), 16)

            if not (0 <= key <= 0xffff and 0 <= iv <= 0xffff):
                raise ValueError
            plain_str = cbc_decrypt(cipher_hex, key, iv)
            self.cbc_result.config(state="normal")
            self.cbc_result.delete("1.0", tk.END)
            self.cbc_result.insert("1.0", plain_str)
            self.cbc_result.config(state="disabled")
        except:
            messagebox.showerror("错误", "请先加密获取密文，或确保输入格式正确")

    def test_tamper(self):
        try:
            idx = int(self.tamper_idx.get())
            result = cbc_tamper_test(
                self.current_cipher_hex,
                int(self.cbc_key.get(), 16),
                self.current_iv,
                idx
            )
            self.tamper_result.config(state="normal")
            self.tamper_result.delete("1.0", tk.END)
            self.tamper_result.insert("1.0", result)
            self.tamper_result.config(state="disabled")
        except:
            messagebox.showerror("错误", "请先进行加密，再输入有效的块索引")


# 测试函数
def run_tests():
    # 基础加密解密测试
    plain = 0x1234
    key = 0x0f15
    cipher = s_aes_encrypt(plain, key)
    decrypted = s_aes_decrypt(cipher, key)
    print(f"基础测试: 明文={plain:04x}, 密钥={key:04x}, 密文={cipher:04x}, 解密后={decrypted:04x}")
    assert decrypted == plain, "基础测试失败"

    # 字符串加密测试
    s = "test"
    key_str = 0x3a5f
    encrypted_hex = encrypt_str(s, key_str)
    decrypted_str = decrypt_str(encrypted_hex, key_str)
    print(f"字符串测试: 原始='{s}', 加密后（十六进制）='{encrypted_hex}', 解密后='{decrypted_str}'")
    assert decrypted_str == s, "字符串测试失败"

    # 双重加密测试
    key32 = 0x12345678
    cipher_double = double_encrypt(plain, key32)
    decrypted_double = double_decrypt(cipher_double, key32)
    print(f"双重加密测试: 密文={cipher_double:04x}, 解密后={decrypted_double:04x}")
    assert decrypted_double == plain, "双重加密测试失败"

    # 三重加密测试
    key48 = 0x123456789abc
    cipher_triple = triple_encrypt(plain, key48)
    decrypted_triple = triple_decrypt(cipher_triple, key48)
    print(f"三重加密测试: 密文={cipher_triple:04x}, 解密后={decrypted_triple:04x}")
    assert decrypted_triple == plain, "三重加密测试失败"

    # CBC模式测试
    cbc_plain = "cbc test"
    cbc_key = 0xabcd
    cbc_cipher_hex, iv = cbc_encrypt(cbc_plain, cbc_key)
    cbc_decrypted = cbc_decrypt(cbc_cipher_hex, cbc_key, iv)
    print(f"CBC测试: 原始='{cbc_plain}', 密文（十六进制）='{cbc_cipher_hex}', 解密后='{cbc_decrypted}'")
    assert cbc_decrypted == cbc_plain, "CBC测试失败"

    print("所有测试通过!")


if __name__ == "__main__":
    run_tests()
    root = tk.Tk()
    app = S_AES_GUI(root)
    root.mainloop()