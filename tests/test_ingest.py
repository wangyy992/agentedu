from tutor.ingest.chunker import chunk_material, split_sections


def test_sections_keep_heading_path():
    text = "# 主标题\n开篇。\n\n## 一节\n内容甲。\n\n### 一点一\n内容乙。\n\n## 二节\n内容丙。"
    sections = dict(split_sections(text))
    assert "主标题 > 一节 > 一点一" in sections
    assert sections["主标题 > 二节"].strip() == "内容丙。"


def test_no_heading_document_still_chunks():
    material = chunk_material("m", "无标题", "就是一段没有任何标题的正文。" * 5)
    assert len(material.chunks) == 1
    assert material.chunks[0].section == "无标题"


def test_chunk_ids_are_ordered_and_unique(sample_text):
    material = chunk_material("m", "t", sample_text)
    ids = [c.id for c in material.chunks]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    assert all(c.text.strip() for c in material.chunks)


def test_long_paragraph_is_split():
    long_para = "这是一个很长的句子,用来测试硬切分逻辑。" * 60
    material = chunk_material("m", "t", long_para)
    assert len(material.chunks) > 1
    assert all(len(c.text) <= 1200 for c in material.chunks)


def test_full_text_carries_chunk_ids(sample_text):
    material = chunk_material("m", "t", sample_text)
    full = material.full_text()
    assert "[c001]" in full and f"[{material.chunks[-1].id}]" in full
