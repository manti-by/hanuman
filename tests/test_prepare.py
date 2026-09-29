from unittest.mock import MagicMock, patch

import pytest

from hanuman.prepare import convert_pdfs_to_markdown, normalize_headings, prepare, split_markdown, strip_page_furniture


class TestStripPageFurniture:
    def test_removes_repeated_running_header(self):
        markdown = "\n".join(f"**RUN-1**\n\nbody {i}" for i in range(10))
        assert "RUN-1" not in strip_page_furniture(markdown)

    def test_removes_standalone_page_numbers(self):
        markdown = "\n".join(f"**RUN-1**\n\n{page}\n\nbody" for page in range(30, 40))
        result = strip_page_furniture(markdown)
        assert "30" not in result.splitlines()

    def test_keeps_short_document_untouched(self):
        markdown = "**ТКП 339-2022**\n\nbody\n\n42"
        assert strip_page_furniture(markdown) == markdown

    def test_keeps_numbers_inside_text(self):
        markdown = "\n".join("**RUN-1**\n\nнапряжением 42 В применяется" for _ in range(6))
        assert "42" in strip_page_furniture(markdown)


class TestNormalizeHeadings:
    def test_derives_level_from_number_depth(self):
        assert normalize_headings("## **4.1  Общие положения**").startswith("## ")
        assert normalize_headings("### **4.2.1 Расчет**").startswith("### ")
        assert normalize_headings("### **4.4.3.3 Испытание**").startswith("#### ")

    def test_top_level_number_becomes_h1(self):
        assert normalize_headings("## **4  Общие правила**").startswith("# ")

    def test_joins_wrapped_heading_continuation(self):
        result = normalize_headings("### **4.4.9.4 Испытание изоляции**\n\n### **частотой 50 Гц:**")
        assert result.strip() == "#### 4.4.9.4 Испытание изоляции частотой 50 Гц:"

    def test_joins_hyphenated_word_without_space(self):
        result = normalize_headings("### **4.4.10.8 Проверка характери-**\n\n### **стик**")
        assert result.strip() == "#### 4.4.10.8 Проверка характеристик"

    def test_keeps_numbered_heading_separate(self):
        result = normalize_headings("### **4.4.3.1 Измерение**\n\n### **4.4.3.2 Проверка**")
        assert result.count("#### ") == 2

    def test_strips_bold_markup_from_heading(self):
        assert normalize_headings("## **4.1  Общие положения**").strip() == "## 4.1 Общие положения"

    def test_demotes_table_caption_to_bold_text(self):
        assert normalize_headings("#### **Таблица 4.4.14 – Напряжения**").strip() == ("**Таблица 4.4.14 – Напряжения**")

    def test_keeps_table_caption_out_of_header_metadata(self):
        markdown = "# 4 Общие правила\n\nbody\n\n#### **Таблица 4.4.14 – Напряжения**\n\n| a | b |"
        result = normalize_headings(markdown)
        assert "# 4 Общие правила" in result
        assert "#### " not in result

    def test_leaves_body_text_untouched(self):
        markdown = "50 В переменного и 120 В постоянного тока."
        assert normalize_headings(markdown) == markdown


class TestSplitMarkdown:
    def test_produces_one_chunk_per_section(self):
        documents = split_markdown("# 4 Общие правила\n\nbody a\n\n## 4.1 Общие положения\n\nbody b", "doc")
        assert len(documents) == 2

    def test_records_header_path_in_metadata(self):
        documents = split_markdown("# 4 Общие\n\nbody\n\n### 4.4.3.3 Испытание\n\nbody", "doc")
        assert documents[-1].metadata["section"] == "4 Общие > 4.4.3.3 Испытание"
        assert documents[-1].metadata["source"] == "doc"

    def test_bounds_oversized_sections(self):
        documents = split_markdown("# 4\n\n" + ("слово " * 5000), "doc")
        assert len(documents) > 1
        assert all(len(document.page_content) <= 1200 for document in documents)

    def test_keeps_metadata_on_every_piece(self):
        documents = split_markdown("# 4\n\n" + ("слово " * 5000), "doc")
        assert {document.metadata["source"] for document in documents} == {"doc"}


class TestPrepare:
    @pytest.mark.asyncio
    @patch("hanuman.prepare.write_chunks", return_value=2)
    @patch("hanuman.prepare.convert_pdfs_to_markdown")
    async def test_prepare_splits_converted_markdown(self, mock_convert, mock_write):
        markdown_file = MagicMock()
        markdown_file.stem = "doc"
        markdown_file.read_text.return_value = "# 4 Общие правила\n\nbody"
        mock_convert.return_value = [markdown_file]

        await prepare()

        mock_write.assert_called_once()
        assert mock_write.call_args[0][1] == "doc"

    @pytest.mark.asyncio
    @patch("hanuman.prepare.convert_pdfs_to_markdown", return_value=[])
    async def test_prepare_exits_early_without_pdfs(self, mock_convert):
        await prepare()

    @pytest.mark.asyncio
    @patch("hanuman.prepare.pymupdf4llm.to_markdown", return_value="# 4 Общие правила")
    async def test_convert_writes_markdown_file(self, mock_to_markdown, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "manual.pdf").write_bytes(b"%PDF-1.4")
        markdown_dir = tmp_path / "markdown"

        with patch("hanuman.prepare.settings") as mock_settings:
            mock_settings.raw_dir = raw_dir
            mock_settings.markdown_dir = markdown_dir
            converted = await convert_pdfs_to_markdown()

        assert converted == [markdown_dir / "manual.md"]
        assert converted[0].read_text(encoding="utf-8") == "# 4 Общие правила"

    @pytest.mark.asyncio
    async def test_convert_without_pdfs_returns_empty(self, tmp_path):
        with patch("hanuman.prepare.settings") as mock_settings:
            mock_settings.raw_dir = tmp_path / "raw"
            assert await convert_pdfs_to_markdown() == []
