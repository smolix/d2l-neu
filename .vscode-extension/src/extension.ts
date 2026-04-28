import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { spawn, ChildProcess } from 'child_process';

import { registerSync } from './sync';
import { registerLint } from './lint';

const FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet'] as const;
type Framework = typeof FRAMEWORKS[number];

let statusBar: vscode.StatusBarItem;
let previewProc: ChildProcess | undefined;

function pyExe(): string {
    return vscode.workspace.getConfiguration('d2l').get<string>('python', 'python3');
}

function workspaceRoot(): string {
    const ws = vscode.workspace.workspaceFolders?.[0];
    if (!ws) {
        throw new Error('No workspace folder open');
    }
    return ws.uri.fsPath;
}

function setStatus(state: 'idle' | 'syncing' | 'conflict' | 'error', tooltip = '') {
    if (!statusBar) return;
    switch (state) {
        case 'idle':
            statusBar.text = '$(check) d2l';
            statusBar.tooltip = tooltip || 'Sync daemon active';
            statusBar.backgroundColor = undefined;
            break;
        case 'syncing':
            statusBar.text = '$(sync~spin) d2l: syncing…';
            statusBar.tooltip = tooltip;
            break;
        case 'conflict':
            statusBar.text = '$(warning) d2l: conflict';
            statusBar.tooltip = tooltip || 'Click for details';
            statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            break;
        case 'error':
            statusBar.text = '$(error) d2l';
            statusBar.tooltip = tooltip;
            statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            break;
    }
}

async function pickFramework(): Promise<Framework | undefined> {
    const choice = await vscode.window.showQuickPick(FRAMEWORKS as unknown as string[], {
        placeHolder: 'Select framework',
    });
    return choice as Framework | undefined;
}

function notebookPathFor(sourceMd: string, fw: Framework): string {
    // chapter_foo/bar.md → _notebooks/<fw>/chapter_foo/bar.ipynb
    const root = workspaceRoot();
    const rel = path.relative(root, sourceMd);
    const ipynb = rel.replace(/\.md$/, '.ipynb');
    return path.join(root, '_notebooks', fw, ipynb);
}

function sourceMdFromNotebook(notebookPath: string): string | undefined {
    // _notebooks/<fw>/<chapter>/<file>.ipynb → <chapter>/<file>.md
    const root = workspaceRoot();
    const rel = path.relative(path.join(root, '_notebooks'), notebookPath);
    const m = rel.match(/^([^/\\]+)[/\\](.+)\.ipynb$/);
    if (!m) return undefined;
    return path.join(root, m[2] + '.md');
}

function frameworkFromNotebook(notebookPath: string): Framework | undefined {
    const root = workspaceRoot();
    const rel = path.relative(path.join(root, '_notebooks'), notebookPath);
    const fw = rel.split(/[/\\]/, 1)[0];
    return (FRAMEWORKS as unknown as string[]).includes(fw) ? (fw as Framework) : undefined;
}

async function ensureNotebook(md: string, fw: Framework, output: vscode.OutputChannel): Promise<string> {
    const ipynb = notebookPathFor(md, fw);
    const root = workspaceRoot();
    const rel = path.relative(root, md);
    const needs = !fs.existsSync(ipynb)
        || fs.statSync(md).mtimeMs > fs.statSync(ipynb).mtimeMs;
    if (needs) {
        output.appendLine(`generating notebook for ${rel} (${fw})`);
        await runPython(['tools/gen_notebooks.py', '.', '_notebooks',
            '--frameworks', fw, '--convert'], root, output);
    }
    return ipynb;
}

async function runPython(args: string[], cwd: string, output: vscode.OutputChannel): Promise<number> {
    return new Promise(resolve => {
        const proc = spawn(pyExe(), args, { cwd });
        proc.stdout.on('data', d => output.append(String(d)));
        proc.stderr.on('data', d => output.append(String(d)));
        proc.on('exit', code => resolve(code ?? 1));
    });
}

async function editFrameworkView(output: vscode.OutputChannel) {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !editor.document.fileName.endsWith('.md')) {
        vscode.window.showErrorMessage('Open a chapter_*/*.md file first');
        return;
    }
    const fw = await pickFramework();
    if (!fw) return;
    try {
        const ipynb = await ensureNotebook(editor.document.fileName, fw, output);
        await vscode.commands.executeCommand('vscode.open', vscode.Uri.file(ipynb));
    } catch (e) {
        vscode.window.showErrorMessage(`Failed to open framework view: ${e}`);
    }
}

async function switchFramework(output: vscode.OutputChannel) {
    const editor = vscode.window.activeNotebookEditor;
    let md: string | undefined;
    if (editor) {
        md = sourceMdFromNotebook(editor.notebook.uri.fsPath);
    } else {
        const text = vscode.window.activeTextEditor?.document.fileName;
        if (text?.endsWith('.md')) md = text;
    }
    if (!md) {
        vscode.window.showErrorMessage('No notebook or .md file active');
        return;
    }
    const fw = await pickFramework();
    if (!fw) return;
    try {
        const ipynb = await ensureNotebook(md, fw, output);
        if (editor) {
            // Close current notebook, open new one
            await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
        }
        await vscode.commands.executeCommand('vscode.open', vscode.Uri.file(ipynb));
    } catch (e) {
        vscode.window.showErrorMessage(`Switch framework failed: ${e}`);
    }
}

async function watchSlides(output: vscode.OutputChannel) {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !editor.document.fileName.endsWith('.md')) {
        vscode.window.showErrorMessage('Open a chapter_*/*.md file first');
        return;
    }
    const fw = await pickFramework();
    if (!fw) return;
    if (previewProc && !previewProc.killed) {
        previewProc.kill();
    }
    const root = workspaceRoot();
    const rel = path.relative(root, editor.document.fileName);
    const port = vscode.workspace.getConfiguration('d2l')
        .get<number>('slidePreview.port', 4444);
    output.appendLine(`watching ${rel} (${fw})`);
    previewProc = spawn(pyExe(),
        ['tools/watch_slides.py', '--fw', fw, '--file', rel, '--port', String(port)],
        { cwd: root });
    previewProc.stdout?.on('data', d => output.append(String(d)));
    previewProc.stderr?.on('data', d => output.append(String(d)));
    previewProc.on('exit', () => {
        previewProc = undefined;
    });
    vscode.env.openExternal(vscode.Uri.parse(`http://localhost:${port}/`));
}

async function revealSourceForCell(output: vscode.OutputChannel) {
    const ed = vscode.window.activeNotebookEditor;
    if (!ed) {
        vscode.window.showErrorMessage('Open a notebook first');
        return;
    }
    const sel = ed.selection;
    if (!sel) return;
    const cell = ed.notebook.cellAt(sel.start);
    const cellId = cell.metadata?.id ?? (cell as any).id;
    if (!cellId) {
        vscode.window.showInformationMessage('Cell has no #<id>; cannot reveal source');
        return;
    }
    const md = sourceMdFromNotebook(ed.notebook.uri.fsPath);
    if (!md || !fs.existsSync(md)) {
        vscode.window.showErrorMessage('Source .md not found');
        return;
    }
    const text = fs.readFileSync(md, 'utf-8');
    const idx = text.indexOf(`#${cellId}`);
    if (idx < 0) {
        vscode.window.showInformationMessage(`#${cellId} not found in source`);
        return;
    }
    const line = text.slice(0, idx).split('\n').length;
    const doc = await vscode.workspace.openTextDocument(md);
    const editor = await vscode.window.showTextDocument(doc);
    const pos = new vscode.Position(line - 1, 0);
    editor.selection = new vscode.Selection(pos, pos);
    editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
}

async function lintSource(output: vscode.OutputChannel, diag: vscode.DiagnosticCollection) {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !editor.document.fileName.endsWith('.md')) return;
    const root = workspaceRoot();
    output.appendLine(`linting ${path.relative(root, editor.document.fileName)}`);
    await runLint(editor.document.fileName, root, diag, output);
}

async function runLint(file: string, root: string, diag: vscode.DiagnosticCollection,
                       output: vscode.OutputChannel) {
    const proc = spawn(pyExe(), ['tools/lint_source.py', file], { cwd: root });
    let stdout = '';
    proc.stdout.on('data', d => { stdout += String(d); });
    proc.stderr.on('data', d => output.append(String(d)));
    return new Promise<void>(resolve => {
        proc.on('exit', () => {
            const issues: vscode.Diagnostic[] = [];
            for (const line of stdout.split('\n')) {
                const m = line.match(/^(.+):(\d+):(\d+): (warning|error): (.+)$/);
                if (!m) continue;
                const severity = m[4] === 'error'
                    ? vscode.DiagnosticSeverity.Error
                    : vscode.DiagnosticSeverity.Warning;
                const ln = parseInt(m[2]) - 1;
                const col = parseInt(m[3]) - 1;
                const range = new vscode.Range(ln, col, ln, col + 1);
                issues.push(new vscode.Diagnostic(range, m[5], severity));
            }
            diag.set(vscode.Uri.file(file), issues);
            resolve();
        });
    });
}

async function openSlidePreview(output: vscode.OutputChannel) {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !editor.document.fileName.endsWith('.md')) return;
    const fw = await pickFramework();
    if (!fw) return;
    const root = workspaceRoot();
    const rel = path.relative(root, editor.document.fileName);
    output.appendLine(`one-shot slide render: ${rel} (${fw})`);
    const code = await runPython(
        ['tools/gen_slides.py', '.', '_slides', '--frameworks', fw, '--render',
         '--workers', '4', '--files', rel],
        root, output);
    if (code !== 0) {
        vscode.window.showErrorMessage('Slide render failed');
        return;
    }
    const html = path.join(root, '_slides', fw, rel.replace(/\.md$/, '.html'));
    if (fs.existsSync(html)) {
        vscode.env.openExternal(vscode.Uri.file(html));
    }
}

let syncEnabled = true;

export function activate(context: vscode.ExtensionContext) {
    statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.command = 'd2l.toggleSyncDaemon';
    setStatus('idle');
    statusBar.show();
    context.subscriptions.push(statusBar);

    const output = vscode.window.createOutputChannel('d2l');
    context.subscriptions.push(output);

    const diag = vscode.languages.createDiagnosticCollection('d2l');
    context.subscriptions.push(diag);

    context.subscriptions.push(
        vscode.commands.registerCommand('d2l.editFrameworkView', () =>
            editFrameworkView(output)),
        vscode.commands.registerCommand('d2l.switchFramework', () =>
            switchFramework(output)),
        vscode.commands.registerCommand('d2l.watchSlides', () =>
            watchSlides(output)),
        vscode.commands.registerCommand('d2l.revealSourceForCell', () =>
            revealSourceForCell(output)),
        vscode.commands.registerCommand('d2l.lintSource', () =>
            lintSource(output, diag)),
        vscode.commands.registerCommand('d2l.openSlidePreview', () =>
            openSlidePreview(output)),
        vscode.commands.registerCommand('d2l.toggleSyncDaemon', () => {
            syncEnabled = !syncEnabled;
            vscode.window.showInformationMessage(
                `d2l sync daemon: ${syncEnabled ? 'on' : 'off'}`);
            setStatus(syncEnabled ? 'idle' : 'error',
                syncEnabled ? 'Sync daemon active' : 'Sync daemon disabled');
        }),
    );

    // Sync daemon: watch _notebooks/<fw>/**/*.ipynb saves
    if (vscode.workspace.getConfiguration('d2l').get<boolean>('syncDaemon.enabled', true)) {
        registerSync(context, output, () => syncEnabled, setStatus,
            (nb) => sourceMdFromNotebook(nb), (nb) => frameworkFromNotebook(nb),
            pyExe, workspaceRoot);
    }

    // Lint on save
    registerLint(context, diag, output, runLint, workspaceRoot);
}

export function deactivate() {
    if (previewProc && !previewProc.killed) {
        previewProc.kill();
    }
}
