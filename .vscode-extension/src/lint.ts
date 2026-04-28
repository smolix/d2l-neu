import * as vscode from 'vscode';
import * as path from 'path';

type LintRunner = (
    file: string, root: string, diag: vscode.DiagnosticCollection,
    output: vscode.OutputChannel,
) => Promise<void>;

export function registerLint(
    context: vscode.ExtensionContext,
    diag: vscode.DiagnosticCollection,
    output: vscode.OutputChannel,
    runLint: LintRunner,
    workspaceRoot: () => string,
) {
    const onSave = async (doc: vscode.TextDocument) => {
        if (!doc.fileName.endsWith('.md')) return;
        const root = workspaceRoot();
        const rel = path.relative(root, doc.fileName);
        if (!rel.startsWith('chapter_')) return;
        await runLint(doc.fileName, root, diag, output);
    };
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument(onSave));
}
