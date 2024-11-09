using Microsoft.VisualStudio.Text;
using Microsoft.VisualStudio.Text.Editor;
using Microsoft.VisualStudio.Utilities;
using System.ComponentModel.Composition;
using System.Diagnostics;

namespace NvAssist
{
    [Command(PackageIds.NvAssist)]
    internal sealed class NvAssist : BaseCommand<NvAssist>
    {
        protected override async Task ExecuteAsync(OleMenuCmdEventArgs e)
        {
            DocumentView docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView?.TextView == null) return;

            SnapshotPoint position = docView.TextView.Caret.Position.BufferPosition;
            docView.TextBuffer?.Insert(position, "some text");

            ITextSnapshot snapshot = docView.TextBuffer.CurrentSnapshot;

            await VS.MessageBox.ShowWarningAsync("NvAssist", snapshot.GetText());
        }
    }
}
