import * as vscode from 'vscode';
import * as path from 'path';
import { spawn } from 'child_process';

type StatusSetter = (
    s: 'idle' | 'syncing' | 'conflict' | 'error', tooltip?: string) => void;

export function registerSync(
    context: vscode.ExtensionContext,
    output: vscode.OutputChannel,
    isEnabled: () => boolean,
    setStatus: StatusSetter,
    sourceFromNotebook: (nb: string) => string | undefined,
    frameworkFromNotebook: (nb: string) => string | undefined,
    pyExe: () => string,
    workspaceRoot: () => string,
) {
    const debounceMs = vscode.workspace.getConfiguration('d2l')
        .get<number>('syncDaemon.debounceMs', 500);
    const pending = new Map<string, NodeJS.Timeout>();

    const onSave = (doc: vscode.NotebookDocument) => {
        if (!isEnabled()) return;
        const nbPath = doc.uri.fsPath;
        const root = workspaceRoot();
        const nbDir = path.join(root, '_notebooks');
        if (!nbPath.startsWith(nbDir)) return;
        const md = sourceFromNotebook(nbPath);
        const fw = frameworkFromNotebook(nbPath);
        if (!md || !fw) return;

        clearTimeout(pending.get(nbPath));
        pending.set(nbPath, setTimeout(() => {
            pending.delete(nbPath);
            setStatus('syncing', `sync_back ${path.basename(nbPath)}`);
            const proc = spawn(pyExe(),
                ['tools/sync_back.py', '--notebook', nbPath, '--source', md,
                 '--framework', fw],
                { cwd: root });
            let buf = '';
            proc.stdout.on('data', d => { buf += String(d); output.append(String(d)); });
            proc.stderr.on('data', d => { buf += String(d); output.append(String(d)); });
            proc.on('exit', code => {
                if (code === 0) {
                    setStatus('idle', `sync_back: ${buf.trim().split('\n').pop()}`);
                } else {
                    setStatus('error', 'sync_back failed; see output');
                    vscode.window.showWarningMessage(
                        'd2l sync_back failed — see output panel');
                }
            });
        }, debounceMs));
    };

    context.subscriptions.push(vscode.workspace.onDidSaveNotebookDocument(onSave));
}
