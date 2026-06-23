from typing import Annotated, Tuple
import sys

# 方法1: 使用类型注解 + 运行时验证
def validate_pair_type(pair: Tuple[bytes, bytes]) -> Tuple[bytes, bytes]:
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError(f"Expected tuple[bytes, bytes], got {type(pair)}")
    if not all(isinstance(x, bytes) for x in pair):
        raise TypeError(f"Expected tuple[bytes, bytes], got {type(pair)}")
    return pair

# 声明时进行类型检查
def declare_pair():
    # 正确的类型
    pair1: Tuple[bytes, bytes] = validate_pair_type((b'h', b'e'))
    print(f"pair1: {pair1}")
    
    # 错误的类型 - 会报错
    try:
        pair2: Tuple[bytes, bytes] = validate_pair_type((b'h',))  # 只有1个bytes
    except TypeError as e:
        print(f"Error: {e}")

# 方法2: 使用 dataclass 或 pydantic
from dataclasses import dataclass

@dataclass
class BytePair:
    first: bytes
    second: bytes
    
    def __post_init__(self):
        if not isinstance(self.first, bytes) or not isinstance(self.second, bytes):
            raise TypeError("Both elements must be bytes")

# 方法3: 使用 pydantic (需要安装: pip install pydantic)
try:
    from pydantic import BaseModel, ValidationError
    
    class BytePairModel(BaseModel):
        first: bytes
        second: bytes
    
    # 正确的类型
    pair3 = BytePairModel(first=b'h', second=b'e')
    print(f"pair3: {pair3}")
    
    # 错误的类型 - 会报错
    try:
        pair4 = BytePairModel(first=b'h')  # 缺少second
    except ValidationError as e:
        print(f"Validation error: {e}")
        
except ImportError:
    print("pydantic not installed, skipping pydantic example")

if __name__ == "__main__":
    print("=== Method 1: Runtime validation ===")
    declare_pair()
    
    print("\n=== Method 2: Dataclass ===")
    try:
        bp1 = BytePair(b'h', b'e')
        print(f"bp1: {bp1}")
        
        bp2 = BytePair(b'h', 101)  # 第二个不是bytes
    except TypeError as e:
        print(f"Error: {e}") 