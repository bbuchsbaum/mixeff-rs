//! Block-structure summaries for blocked mixed-model systems.

use std::fmt;

use crate::model::{GeneralizedLinearMixedModel, LinearMixedModel};
use crate::types::{matrix_block::block_index, MatrixBlock};

/// Description of the blocked `A`/`L` structure for a mixed model.
#[derive(Debug, Clone)]
pub struct BlockDescription {
    /// Block names, typically grouping factors followed by `fixed`.
    pub block_names: Vec<String>,
    /// Row count for each diagonal block.
    pub block_rows: Vec<usize>,
    /// Lower-triangular block type descriptions.
    pub al_types: Vec<Vec<String>>,
}

impl BlockDescription {
    /// Construct a block description from a linear mixed model.
    pub fn from_linear_model(model: &LinearMixedModel) -> Self {
        let mut block_names = model
            .reterms
            .iter()
            .map(|rt| rt.grouping_name.clone())
            .collect::<Vec<_>>();
        block_names.push("fixed".to_string());

        let k = block_names.len();
        let mut block_rows = Vec::with_capacity(k);
        let mut al_types = vec![vec![String::new(); k]; k];

        for i in 0..k {
            block_rows.push(model.a_blocks[block_index(i, i)].nrows());
            for j in 0..=i {
                al_types[i][j] = short_type(
                    &model.a_blocks[block_index(i, j)],
                    &model.l_blocks[block_index(i, j)],
                )
                .to_string();
            }
        }

        Self {
            block_names,
            block_rows,
            al_types,
        }
    }

    /// Construct a block description from a generalized linear mixed model.
    pub fn from_generalized_model(model: &GeneralizedLinearMixedModel) -> Self {
        Self::from_linear_model(&model.lmm)
    }

    /// Render the description as a markdown table.
    pub fn to_markdown(&self) -> String {
        let rows = self.table_rows();
        let widths = column_widths(&rows);
        let mut out = String::new();

        out.push_str(&format_markdown_row(&rows[0], &widths));
        out.push('\n');
        out.push_str(&format_markdown_alignment(&widths));
        out.push('\n');

        for row in rows.iter().skip(1) {
            out.push_str(&format_markdown_row(row, &widths));
            out.push('\n');
        }

        out
    }

    /// Render the description as an HTML table.
    pub fn to_html(&self) -> String {
        let rows = self.table_rows();
        let mut out = String::from("<table><tr>");

        for cell in &rows[0] {
            out.push_str(&format!("<th align=\"left\">{cell}</th>"));
        }
        out.push_str("</tr>");

        for row in rows.iter().skip(1) {
            out.push_str("<tr>");
            for cell in row {
                out.push_str(&format!("<td align=\"left\">{cell}</td>"));
            }
            out.push_str("</tr>");
        }

        out.push_str("</table>\n");
        out
    }

    /// Render the description as a LaTeX table.
    pub fn to_latex(&self) -> String {
        let rows = self.table_rows();
        let mut out = String::new();
        let cols = "l | ".repeat(rows[0].len() - 1) + "l";

        out.push_str("\\begin{tabular}\n");
        out.push_str(&format!("{{{cols}}}\n"));
        out.push_str(&rows[0].join(" & "));
        out.push_str(" \\\\\n");
        out.push_str("\\hline\n");

        for row in rows.iter().skip(1) {
            out.push_str(&row.join(" & "));
            out.push_str(" \\\\\n");
        }

        out.push_str("\\end{tabular}\n");
        out
    }

    fn table_rows(&self) -> Vec<Vec<String>> {
        let mut rows = Vec::with_capacity(self.block_rows.len() + 1);
        let mut header = vec!["rows".to_string()];
        header.extend(self.block_names.iter().cloned());
        rows.push(header);

        for (i, row_count) in self.block_rows.iter().enumerate() {
            let mut row = vec![row_count.to_string()];
            for j in 0..self.block_names.len() {
                if j <= i {
                    row.push(self.al_types[i][j].clone());
                } else {
                    row.push(String::new());
                }
            }
            rows.push(row);
        }

        rows
    }
}

impl fmt::Display for BlockDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let row_width = self
            .block_rows
            .iter()
            .map(|rows| rows.to_string().len())
            .max()
            .unwrap_or(4)
            .max(4)
            + 1;
        let col_width = self
            .block_names
            .iter()
            .map(|name| name.len())
            .max()
            .unwrap_or(4)
            .max(13)
            + 1;

        write!(f, "{:<row_width$}", "rows:")?;
        for name in &self.block_names {
            write!(f, "{:^col_width$}", name)?;
        }
        writeln!(f)?;

        for (i, row_count) in self.block_rows.iter().enumerate() {
            write!(f, "{:>width$}", format!("{row_count}:"), width = row_width)?;
            for j in 0..=i {
                write!(f, "{:^col_width$}", self.al_types[i][j])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

fn short_type(a: &MatrixBlock, l: &MatrixBlock) -> &'static str {
    match (a, l) {
        (MatrixBlock::BlockDiagonal(_), MatrixBlock::BlockDiagonal(_)) => "BlkDiag",
        (MatrixBlock::BlockDiagonal(_), MatrixBlock::Dense(_)) => "BlkDiag/Dense",
        (MatrixBlock::Diagonal(_), MatrixBlock::Diagonal(_)) => "Diagonal",
        (MatrixBlock::Diagonal(_), MatrixBlock::Dense(_)) => "Diag/Dense",
        (MatrixBlock::Dense(_), MatrixBlock::Dense(_)) => "Dense",
        (MatrixBlock::Dense(_), MatrixBlock::Diagonal(_)) => "Dense/Diag",
        (MatrixBlock::Dense(_), MatrixBlock::BlockDiagonal(_)) => "Dense/BlkDiag",
        (MatrixBlock::Dense(_), MatrixBlock::Sparse(_)) => "Dense/Sparse",
        (MatrixBlock::Sparse(_), MatrixBlock::Dense(_)) => "Sparse/Dense",
        (MatrixBlock::Sparse(_), MatrixBlock::Diagonal(_)) => "Sparse/Diag",
        (MatrixBlock::Sparse(_), MatrixBlock::BlockDiagonal(_)) => "Sparse/BlkDiag",
        (MatrixBlock::Sparse(_), MatrixBlock::Sparse(_)) => "Sparse",
        (MatrixBlock::Diagonal(_), MatrixBlock::BlockDiagonal(_)) => "Diag/BlkDiag",
        (MatrixBlock::Diagonal(_), MatrixBlock::Sparse(_)) => "Diag/Sparse",
        (MatrixBlock::BlockDiagonal(_), MatrixBlock::Diagonal(_)) => "BlkDiag/Diag",
        (MatrixBlock::BlockDiagonal(_), MatrixBlock::Sparse(_)) => "BlkDiag/Sparse",
    }
}

fn column_widths(rows: &[Vec<String>]) -> Vec<usize> {
    (0..rows[0].len())
        .map(|col| rows.iter().map(|row| row[col].len()).max().unwrap_or(0))
        .collect()
}

fn format_markdown_row(row: &[String], widths: &[usize]) -> String {
    let mut out = String::new();
    for (cell, width) in row.iter().zip(widths.iter()) {
        out.push_str(&format!("| {:<width$} ", cell, width = *width));
    }
    out.push('|');
    out
}

fn format_markdown_alignment(widths: &[usize]) -> String {
    let mut out = String::new();
    for width in widths {
        out.push_str("|:");
        out.push_str(&"-".repeat(*width));
        out.push(' ');
    }
    out.push('|');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_block_description() -> BlockDescription {
        BlockDescription {
            block_names: vec!["subj".to_string(), "item".to_string(), "fixed".to_string()],
            block_rows: vec![316, 24, 7],
            al_types: vec![
                vec!["Diagonal".to_string(), String::new(), String::new()],
                vec!["Dense".to_string(), "Diag/Dense".to_string(), String::new()],
                vec![
                    "Dense".to_string(),
                    "Dense".to_string(),
                    "Dense".to_string(),
                ],
            ],
        }
    }

    #[test]
    fn test_markdown_matches_julia_example() {
        assert_eq!(
            sample_block_description().to_markdown(),
            concat!(
                "| rows | subj     | item       | fixed |\n",
                "|:---- |:-------- |:---------- |:----- |\n",
                "| 316  | Diagonal |            |       |\n",
                "| 24   | Dense    | Diag/Dense |       |\n",
                "| 7    | Dense    | Dense      | Dense |\n"
            )
        );
    }

    #[test]
    fn test_html_matches_julia_example() {
        assert_eq!(
            sample_block_description().to_html(),
            concat!(
                "<table><tr><th align=\"left\">rows</th><th align=\"left\">subj</th><th align=\"left\">item</th><th align=\"left\">fixed</th></tr>",
                "<tr><td align=\"left\">316</td><td align=\"left\">Diagonal</td><td align=\"left\"></td><td align=\"left\"></td></tr>",
                "<tr><td align=\"left\">24</td><td align=\"left\">Dense</td><td align=\"left\">Diag/Dense</td><td align=\"left\"></td></tr>",
                "<tr><td align=\"left\">7</td><td align=\"left\">Dense</td><td align=\"left\">Dense</td><td align=\"left\">Dense</td></tr></table>\n"
            )
        );
    }

    #[test]
    fn test_latex_matches_julia_example() {
        assert_eq!(
            sample_block_description().to_latex(),
            concat!(
                "\\begin{tabular}\n",
                "{l | l | l | l}\n",
                "rows & subj & item & fixed \\\\\n",
                "\\hline\n",
                "316 & Diagonal &  &  \\\\\n",
                "24 & Dense & Diag/Dense &  \\\\\n",
                "7 & Dense & Dense & Dense \\\\\n",
                "\\end{tabular}\n"
            )
        );
    }

    #[test]
    fn test_plaintext_includes_core_layout() {
        let out = sample_block_description().to_string();

        assert!(out.contains("rows:"));
        assert!(out.contains("subj"));
        assert!(out.contains("item"));
        assert!(out.contains("fixed"));
        assert!(out.contains("316:"));
        assert!(out.contains("Diag/Dense"));
    }
}
