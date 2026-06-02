"""
数据集加载器模块
支持加载 Weibo21 和 FakeNewsDataset 两个数据集
"""
import pickle
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import re


@dataclass
class NewsItem:
    """新闻数据项"""
    text: str              # 新闻文本内容
    label: Optional[str] = None      # 标签（'fake'/'real' 或 0/1）
    id: Optional[str] = None         # 新闻ID
    title: Optional[str] = None      # 新闻标题
    subject: Optional[str] = None    # 新闻主题
    date: Optional[str] = None       # 发布日期
    metadata: Optional[Dict] = None  # 其他元数据


class DatasetLoader:
    """数据集加载器基类"""
    
    def load(self, path: Union[str, Path]) -> List[NewsItem]:
        """
        加载数据集
        
        Args:
            path: 数据集文件路径
            
        Returns:
            List[NewsItem]: 新闻数据项列表
        """
        raise NotImplementedError
    
    def get_texts(self, path: Union[str, Path]) -> List[str]:
        """
        仅获取新闻文本列表
        
        Args:
            path: 数据集文件路径
            
        Returns:
            List[str]: 新闻文本列表
        """
        items = self.load(path)
        return [item.text for item in items]


class Weibo21Loader(DatasetLoader):
    """Weibo21 数据集加载器
    
    Weibo21数据集通常以pkl格式存储，包含训练集、验证集和测试集
    数据格式：通常是字典列表，每个字典包含 'text' 和 'label' 字段
    """
    
    def load(self, path: Union[str, Path]) -> List[NewsItem]:
        """
        加载Weibo21数据集（pkl格式）
        
        Args:
            path: pkl文件路径
            
        Returns:
            List[NewsItem]: 新闻数据项列表
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {path}")
        
        print(f"正在加载 Weibo21 数据集: {path}")
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            items = []
            
            # 处理不同的数据格式
            if isinstance(data, list):
                # 如果是列表格式
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # 字典格式：{'text': ..., 'label': ..., ...}
                        text = item.get('text', item.get('content', item.get('news', '')))
                        label = item.get('label', item.get('label_id', None))
                        
                        # 处理标签格式（可能是0/1或'fake'/'real'）
                        if label is not None:
                            if isinstance(label, int):
                                label = 'fake' if label == 1 else 'real'
                            elif isinstance(label, str):
                                label = label.lower()
                        
                        items.append(NewsItem(
                            text=str(text),
                            label=label,
                            id=item.get('id', str(idx)),
                            metadata=item
                        ))
                    elif isinstance(item, str):
                        # 纯文本格式
                        items.append(NewsItem(text=item, id=str(idx)))
                    else:
                        # 其他格式，尝试转换为字符串
                        items.append(NewsItem(text=str(item), id=str(idx)))
            
            elif isinstance(data, dict):
                # 如果是字典格式，可能包含 'train', 'test' 等键
                for key, value in data.items():
                    if isinstance(value, list):
                        for idx, item in enumerate(value):
                            if isinstance(item, dict):
                                text = item.get('text', item.get('content', ''))
                                label = item.get('label', None)
                                if label is not None and isinstance(label, int):
                                    label = 'fake' if label == 1 else 'real'
                                items.append(NewsItem(
                                    text=str(text),
                                    label=label,
                                    id=f"{key}_{idx}",
                                    metadata=item
                                ))
            
            print(f"成功加载 {len(items)} 条新闻")
            return items
            
        except Exception as e:
            raise ValueError(f"加载Weibo21数据集失败: {e}")
    
    def load_split(self, train_path: Optional[Union[str, Path]] = None,
                   val_path: Optional[Union[str, Path]] = None,
                   test_path: Optional[Union[str, Path]] = None) -> Dict[str, List[NewsItem]]:
        """
        加载Weibo21数据集的训练/验证/测试集
        
        Args:
            train_path: 训练集路径
            val_path: 验证集路径
            test_path: 测试集路径
            
        Returns:
            Dict[str, List[NewsItem]]: 包含 'train', 'val', 'test' 键的字典
        """
        result = {}
        
        if train_path:
            result['train'] = self.load(train_path)
        if val_path:
            result['val'] = self.load(val_path)
        if test_path:
            result['test'] = self.load(test_path)
        
        return result


class FakeNewsDatasetLoader(DatasetLoader):
    """FakeNewsDataset 数据集加载器
    
    FakeNewsDataset通常以CSV格式存储
    包含字段：title, text, subject, date, label
    """
    
    def load(self, path: Union[str, Path], 
             text_column: str = 'text',
             label_column: str = 'label',
             title_column: str = 'title',
             subject_column: str = 'subject',
             date_column: str = 'date') -> List[NewsItem]:
        """
        加载FakeNewsDataset（CSV格式）
        
        Args:
            path: CSV文件路径
            text_column: 文本列名（默认'text'）
            label_column: 标签列名（默认'label'）
            title_column: 标题列名（默认'title'）
            subject_column: 主题列名（默认'subject'）
            date_column: 日期列名（默认'date'）
            
        Returns:
            List[NewsItem]: 新闻数据项列表
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {path}")

        # If a directory is provided, try folder-based loading (e.g., fake/legit subfolders of raw FakeNewsDataset)
        if path.is_dir():
            return self._load_from_folder(path)
        
        print(f"正在加载 FakeNewsDataset: {path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(path, encoding='utf-8')
            
            # 如果utf-8失败，尝试其他编码
            if df.empty or df.isna().all().all():
                df = pd.read_csv(path, encoding='gbk')
            
            items = []
            
            for idx, row in df.iterrows():
                # 获取文本内容
                text = str(row.get(text_column, ''))
                
                # 如果没有text列，尝试使用title列
                if not text or text == 'nan':
                    text = str(row.get(title_column, ''))
                
                # 如果仍然为空，跳过
                if not text or text == 'nan':
                    continue
                
                # 获取标签
                label = row.get(label_column, None)
                if pd.notna(label):
                    label = str(label).lower()
                    # 标准化标签格式
                    if label in ['0', 'false', '0.0']:
                        label = 'real'
                    elif label in ['1', 'true', '1.0']:
                        label = 'fake'
                else:
                    label = None
                
                # 获取其他字段
                title = row.get(title_column, None)
                if pd.notna(title):
                    title = str(title)
                else:
                    title = None
                
                subject = row.get(subject_column, None)
                if pd.notna(subject):
                    subject = str(subject)
                else:
                    subject = None
                
                date = row.get(date_column, None)
                if pd.notna(date):
                    date = str(date)
                else:
                    date = None
                
                # 创建NewsItem
                items.append(NewsItem(
                    text=text,
                    label=label,
                    id=str(row.get('id', idx)),
                    title=title,
                    subject=subject,
                    date=date,
                    metadata=row.to_dict()
                ))
            
            print(f"成功加载 {len(items)} 条新闻")
            return items
            
        except Exception as e:
            raise ValueError(f"加载FakeNewsDataset失败: {e}")

    def _load_from_folder(self, base_dir: Path) -> List[NewsItem]:
        """
        Load from a folder structure with subfolders such as 'fake' and 'legit'.
        Expected layout (case-insensitive):
            base_dir/
                fake/*.txt
                legit/*.txt
        """
        print(f"检测到目录模式，正在遍历: {base_dir}")
        fake_names = {"fake", "rumor", "fraud"}
        real_names = {"real", "legit", "true", "nonrumor", "non-rumor"}

        items: List[NewsItem] = []
        for sub in base_dir.iterdir():
            if not sub.is_dir():
                continue
            name_lower = sub.name.lower()
            if name_lower in fake_names:
                label = "fake"
            elif name_lower in real_names:
                label = "real"
            else:
                # skip unrelated folders
                continue

            for f in sorted(sub.glob("**/*")):
                if f.is_dir():
                    continue
                try:
                    text = f.read_text(encoding="utf-8").strip()
                except UnicodeDecodeError:
                    text = f.read_text(encoding="latin-1", errors="ignore").strip()
                if not text:
                    continue

                # Try to infer subject from filename prefix (letters before digits)
                subject_match = re.match(r"([A-Za-z]+)", f.stem)
                subject = subject_match.group(1).lower() if subject_match else None

                items.append(
                    NewsItem(
                        text=text,
                        label=label,
                        id=f.name,
                        title=None,
                        subject=subject,
                        date=None,
                        metadata={"path": str(f)}
                    )
                )

        print(f"成功加载 {len(items)} 条新闻 (目录模式)")
        return items


class StrategyQALoader(DatasetLoader):
    """StrategyQA processed JSONL loader.

    Expected input is generated by StrategyQA/prepare_strategyqa.py, with one
    JSON object per line containing id, text, label, task_type, and metadata.
    Labels are expected to be normalized to YES / NO.
    """

    VALID_LABELS = {"YES", "NO"}

    def load(self, path: Union[str, Path]) -> List[NewsItem]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {path}")
        if path.is_dir():
            raise ValueError(f"StrategyQA loader expects a JSONL file, got directory: {path}")

        print(f"正在加载 StrategyQA processed 数据集: {path}")

        items: List[NewsItem] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if not text:
                        continue

                    label = row.get("label")
                    label = str(label).strip().upper() if label is not None else None
                    if label is not None and label not in self.VALID_LABELS:
                        raise ValueError(f"第 {idx} 行 StrategyQA label 无效: {label}")

                    metadata = row.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        metadata = {"raw_metadata": metadata}
                    metadata = {
                        **metadata,
                        "task_type": row.get("task_type", "strategyqa"),
                        "source_path": str(path),
                    }

                    items.append(
                        NewsItem(
                            text=text,
                            label=label,
                            id=str(row.get("id", f"strategyqa_{idx:05d}")),
                            subject="strategyqa",
                            metadata=metadata,
                        )
                    )

            print(f"成功加载 {len(items)} 条 StrategyQA 样本")
            return items
        except json.JSONDecodeError as e:
            raise ValueError(f"加载StrategyQA失败: JSONL 第 {idx} 行格式错误: {e}") from e
        except Exception as e:
            raise ValueError(f"加载StrategyQA失败: {e}") from e


class PubMedQALoader(DatasetLoader):
    """PubMedQA processed JSONL loader.

    Expected input is generated by PubMedQA/prepare_pubmedqa.py, with one JSON
    object per line containing id, text, label, task_type, and metadata. Labels
    are expected to be normalized to YES / NO / MAYBE.
    """

    VALID_LABELS = {"YES", "NO", "MAYBE"}
    BANNED_TEXT_MARKERS = {
        "final_decision",
        "LONG_ANSWER",
        "reasoning_required_pred",
        "reasoning_free_pred",
    }

    def load(self, path: Union[str, Path]) -> List[NewsItem]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {path}")
        if path.is_dir():
            raise ValueError(f"PubMedQA loader expects a JSONL file, got directory: {path}")

        print(f"正在加载 PubMedQA processed 数据集: {path}")

        items: List[NewsItem] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    row = json.loads(line)
                    text = str(row.get("text", "")).strip()
                    if not text:
                        continue

                    leakage_markers = [marker for marker in self.BANNED_TEXT_MARKERS if marker in text]
                    if leakage_markers:
                        raise ValueError(
                            f"第 {idx} 行 PubMedQA text 可能包含泄漏字段: {leakage_markers}"
                        )

                    label = row.get("label")
                    label = str(label).strip().upper() if label is not None else None
                    if label is not None and label not in self.VALID_LABELS:
                        raise ValueError(f"第 {idx} 行 PubMedQA label 无效: {label}")

                    metadata = row.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        metadata = {"raw_metadata": metadata}
                    metadata = {
                        **metadata,
                        "task_type": row.get("task_type", "pubmedqa"),
                        "source_path": str(path),
                    }

                    items.append(
                        NewsItem(
                            text=text,
                            label=label,
                            id=str(row.get("id", f"pubmedqa_{idx:05d}")),
                            subject="pubmedqa",
                            metadata=metadata,
                        )
                    )

            print(f"成功加载 {len(items)} 条 PubMedQA 样本")
            return items
        except json.JSONDecodeError as e:
            raise ValueError(f"加载PubMedQA失败: JSONL 第 {idx} 行格式错误: {e}") from e
        except Exception as e:
            raise ValueError(f"加载PubMedQA失败: {e}") from e


def load_dataset(dataset_type: str, path: Union[str, Path], **kwargs) -> List[NewsItem]:
    """
    统一的数据集加载接口
    
    Args:
        dataset_type: 数据集类型 ('weibo21', 'fakenewsdataset', 或 'strategyqa')
        path: 数据集文件路径
        **kwargs: 其他参数（传递给具体的加载器）
        
    Returns:
        List[NewsItem]: 新闻数据项列表
        
    Example:
        # 加载Weibo21数据集
        items = load_dataset('weibo21', 'data/weibo21/test.pkl')
        
        # 加载FakeNewsDataset
        items = load_dataset('fakenewsdataset', 'data/fakenewsdataset.csv')
    """
    dataset_type = dataset_type.lower()
    
    if dataset_type == 'weibo21':
        loader = Weibo21Loader()
        return loader.load(path)
    elif dataset_type in ['fakenewsdataset', 'fakenews', 'fake_news']:
        loader = FakeNewsDatasetLoader()
        return loader.load(path, **kwargs)
    elif dataset_type in ['strategyqa', 'strategy_qa']:
        loader = StrategyQALoader()
        return loader.load(path)
    elif dataset_type in ['pubmedqa', 'pubmed_qa']:
        loader = PubMedQALoader()
        return loader.load(path)
    else:
        raise ValueError(
            f"不支持的数据集类型: {dataset_type}。"
            "支持的类型: 'weibo21', 'fakenewsdataset', 'strategyqa', 'pubmedqa'"
        )


if __name__ == "__main__":
    # 测试代码
    print("数据集加载器测试")
    print("=" * 50)
    
    # 示例：加载Weibo21数据集
    # items = load_dataset('weibo21', 'path/to/weibo21/test.pkl')
    # print(f"加载了 {len(items)} 条新闻")
    # if items:
    #     print(f"第一条新闻: {items[0].text[:100]}...")
    #     print(f"标签: {items[0].label}")
    
    # 示例：加载FakeNewsDataset
    # items = load_dataset('fakenewsdataset', 'path/to/fakenewsdataset.csv')
    # print(f"加载了 {len(items)} 条新闻")
    # if items:
    #     print(f"第一条新闻: {items[0].text[:100]}...")
    #     print(f"标题: {items[0].title}")
    #     print(f"标签: {items[0].label}")
    
    pass

