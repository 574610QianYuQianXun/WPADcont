from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# 1. 生成 RSA 密钥对（2048 位）
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()

# 2. 将公钥私钥保存为 PEM 格式（字符串形式）
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

print("🗝️ 私钥（PEM 格式）:")
print(pem_private.decode())

print("🔓 公钥（PEM 格式）:")
print(pem_public.decode())

# 3. 加密一段消息
message = "秘密信息：12345".encode("utf-8")

ciphertext = public_key.encrypt(
    message,
    padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                 algorithm=hashes.SHA256(),
                 label=None)
)
print("\n📦 加密后的密文（bytes）:", ciphertext)

# 4. 解密
plaintext = private_key.decrypt(
    ciphertext,
    padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                 algorithm=hashes.SHA256(),
                 label=None)
)
print("✅ 解密结果:", plaintext.decode("utf-8"))
