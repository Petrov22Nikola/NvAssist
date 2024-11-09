global using Community.VisualStudio.Toolkit;
global using Microsoft.VisualStudio.Shell;
global using System;
global using Task = System.Threading.Tasks.Task;
using System.Runtime.InteropServices;
using System.Threading;

using Microsoft.VisualStudio.Text;
using Microsoft.VisualStudio.Text.Editor;
using Microsoft.VisualStudio.Utilities;
using System.ComponentModel.Composition;

using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.IO;
using Microsoft.VisualStudio.Text.Classification;
using System.Windows.Media;
using System.Windows;
using System.Collections.Generic;
using System.Windows.Media.TextFormatting;
using System.Linq;

[Export(typeof(IWpfTextViewCreationListener))]
[ContentType("text")]
[TextViewRole(PredefinedTextViewRoles.Editable)]
internal class TextViewCreationListener : IWpfTextViewCreationListener
{
    async Task<string> QueryOllama(string text)
    {
        using (var client = new HttpClient())
        {
            var payload = new
            {
                prompt = text,
                model = "codegemma:2b",
                options = new
                {
                    max_length = 128,
                    stop = new[] { "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_separator|>" }
                }
            };

            var jsonPayload = JsonConvert.SerializeObject(payload);
            var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("http://localhost:11434/api/generate", content);

            if (response.IsSuccessStatusCode)
            {
                return await StreamJsonResponseAsync(response);
            }
            else
            {
                throw new Exception("Failed to get response from Ollama");
            }
        }
    }

    private async Task<string> StreamJsonResponseAsync(HttpResponseMessage response)
    {
        var sentenceBuilder = new StringBuilder();
        int tokenCount = 0; // Track token count

        using (var stream = await response.Content.ReadAsStreamAsync())
        using (var streamReader = new StreamReader(stream))
        {
            string line;
            while ((line = await streamReader.ReadLineAsync()) != null)
            {
                try
                {
                    var jsonObject = JObject.Parse(line);
                    var responseValue = jsonObject["response"]?.ToString();

                    if (!string.IsNullOrEmpty(responseValue))
                    {
                        sentenceBuilder.Append(responseValue);
                    }
                }
                catch (JsonReaderException ex)
                {
                    Console.WriteLine($"Error parsing JSON: {ex.Message}");
                }
            }
        }

        return sentenceBuilder.ToString();
    }

    async void Message(string text)
    {
        await VS.MessageBox.ShowWarningAsync("NvAssist", text);
    }

    private string GetFimPrefix(ITextSnapshotLine currentLine)
    {
        var snapshot = currentLine.Snapshot;
        int startLineNumber = Math.Max(currentLine.LineNumber - 5, 0);

        var lines = new List<string>();
        for (int i = startLineNumber; i <= currentLine.LineNumber; ++i)
        {
            lines.Add(snapshot.GetLineFromLineNumber(i).GetText());
        }

        return string.Join("\n", lines);
    }

    private string GetFimSuffix(ITextSnapshotLine currentLine)
    {
        var snapshot = currentLine.Snapshot;
        int endLineNumber = Math.Min(currentLine.LineNumber + 5, snapshot.LineCount - 1);

        var lines = new List<string>();
        for (int i = currentLine.LineNumber + 1; i <= endLineNumber; i++)
        {
            lines.Add(snapshot.GetLineFromLineNumber(i).GetText());
        }

        return string.Join("\n", lines);
    }

    int numShifts = 0, activationThreshold = 2;

    private async void OnKeyDown(object sender, System.Windows.Input.KeyEventArgs e)
    {
        if (e.Key == System.Windows.Input.Key.LeftShift) ++numShifts;
        else numShifts = 0;

        if (numShifts == activationThreshold)
        {
            DocumentView docView = await VS.Documents.GetActiveDocumentViewAsync();
            if (docView?.TextView == null) return;

            SnapshotPoint pos = docView.TextView.Caret.Position.BufferPosition;
            ITextSnapshot ss = docView.TextBuffer.CurrentSnapshot;
            ITextSnapshotLine currentLine = ss.GetLineFromPosition(pos);

            string fimPrefix = GetFimPrefix(currentLine);
            string fimSuffix = GetFimSuffix(currentLine);

            string fimPrompt = $"<|fim_prefix|>{fimPrefix}<|fim_suffix|>{fimSuffix}<|fim_middle|>";
            Message("Query: " + fimPrompt);

            string llmResponse = await QueryOllama(fimPrompt);
            Message("LLM Response: " + llmResponse);

            using (var edit = docView.TextBuffer.CreateEdit())
            {
                // Insert the suggestion directly at the caret's current position
                edit.Insert(pos, llmResponse);
                edit.Apply();

                // Move the caret to the end of the inserted text
                var newCaretPosition = pos.Position + llmResponse.Length;
                docView.TextView.Caret.MoveTo(new SnapshotPoint(docView.TextBuffer.CurrentSnapshot, newCaretPosition));
            }
            numShifts = 0;
        }
    }

    public void TextViewCreated(IWpfTextView textView)
    {
        textView.VisualElement.KeyDown += OnKeyDown;
    }
}

namespace NvAssist
{
    [PackageRegistration(UseManagedResourcesOnly = true, AllowsBackgroundLoading = true)]
    [InstalledProductRegistration(Vsix.Name, Vsix.Description, Vsix.Version)]
    [ProvideMenuResource("Menus.ctmenu", 1)]
    [Guid(PackageGuids.NvAssistString)]
    public sealed class NvAssistPackage : ToolkitPackage
    {
        protected override async Task InitializeAsync(CancellationToken cancellationToken, IProgress<ServiceProgressData> progress)
        {
            await this.RegisterCommandsAsync();
            TextViewCreationListener changeListener;
        }
    }
}