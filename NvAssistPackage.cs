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
using System.Reflection;
using System.Diagnostics;
using System.Windows.Input;
using Microsoft.VisualStudio.Shell.Interop;

[Export(typeof(IWpfTextViewCreationListener))]
[ContentType("text")]
[TextViewRole(PredefinedTextViewRoles.Editable)]
internal class TextViewCreationListener : IWpfTextViewCreationListener
{
    async Task<string> QueryQwenFinetuned(string text)
    {
        Mouse.OverrideCursor = Cursors.Wait;
        using (var client = new HttpClient())
        {
            var payload = new { prompt = text };

            var jsonPayload = JsonConvert.SerializeObject(payload);
            var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            // Query the local FastAPI server
            var response = await client.PostAsync("http://localhost:8000/generate", content);
            Mouse.OverrideCursor = null;

            if (response.IsSuccessStatusCode)
            {
                string vuidCheck = await response.Content.ReadAsStringAsync();
                vuidCheck = vuidCheck.Trim('"');
                return vuidCheck;
            }
            else
            {
                throw new Exception("Failed to get response from Hugging Face model");
            }
        }
    }

    async Task<string> QueryGemma(string text)
    {
        Mouse.OverrideCursor = Cursors.Wait;
        using (var client = new HttpClient())
        {
            var payload = new { prompt = text };

            var jsonPayload = JsonConvert.SerializeObject(payload);
            var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            // Query the local FastAPI server
            var response = await client.PostAsync("http://localhost:8000/autocomplete", content);
            Mouse.OverrideCursor = null;

            if (response.IsSuccessStatusCode)
            {
                string autoCode = await response.Content.ReadAsStringAsync();
                autoCode = autoCode.Trim('"');
                return autoCode;
            }
            else
            {
                throw new Exception("Failed to get response from Hugging Face model");
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
        int startLineNumber = Math.Max(currentLine.LineNumber - 10, 0);

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
        int endLineNumber = Math.Min(currentLine.LineNumber + 10, snapshot.LineCount - 1);

        var lines = new List<string>();
        for (int i = currentLine.LineNumber + 1; i <= endLineNumber; i++)
        {
            lines.Add(snapshot.GetLineFromLineNumber(i).GetText());
        }

        return string.Join("\n", lines);
    }

    int numShifts = 0, numCaps = 0, shiftThreshold = 2, capsThreshold = 2;
    private DateTime lastCapsLockTime = DateTime.MinValue;
    private DateTime lastShiftTime = DateTime.MinValue;

    private async void OnKeyDown(object sender, System.Windows.Input.KeyEventArgs e)
    {
        var currentTime = DateTime.Now;

        if (e.Key == System.Windows.Input.Key.LeftShift)
        {
            if (currentTime - lastShiftTime < TimeSpan.FromSeconds(0.3))
            {
                ++numShifts;
                lastShiftTime = currentTime;
            }
            else
            {
                numShifts = 1;
                lastShiftTime = currentTime;
            }
        }
        else
        {
            numShifts = 0;
        }

        if (e.Key == System.Windows.Input.Key.CapsLock)
        {
            if (currentTime - lastCapsLockTime < TimeSpan.FromSeconds(0.3))
            {
                ++numCaps;
                lastCapsLockTime = currentTime;
            }
            else
            {
                numCaps = 1;
                lastCapsLockTime = currentTime;
            }
        }
        else
        {
            numCaps = 0;
        }

        if (numCaps != capsThreshold && numShifts != shiftThreshold) return;

        DocumentView docView = await VS.Documents.GetActiveDocumentViewAsync();
        if (docView?.TextView == null) return;

        SnapshotPoint pos = docView.TextView.Caret.Position.BufferPosition;
        ITextSnapshot ss = docView.TextBuffer.CurrentSnapshot;
        ITextSnapshotLine currentLine = ss.GetLineFromPosition(pos);

        if (numShifts == shiftThreshold)
        {
            string fimPrefix = GetFimPrefix(currentLine);
            string fimSuffix = GetFimSuffix(currentLine);

            string fimPrompt = $"<|fim_prefix|>{fimPrefix}<|fim_suffix|>{fimSuffix}<|fim_middle|>";
            //Message("Query: " + fimPrompt);

            string llmResponse = await QueryGemma(fimPrompt);
            llmResponse = llmResponse.Replace("\\n", "\n");
            string[] stopTokens = { "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_separator|>" };
            foreach (var stopToken in stopTokens)
            {
                llmResponse = llmResponse.Replace(stopToken, string.Empty);
            }

            //Message("LLM Response: " + llmResponse);

            using (var edit = docView.TextBuffer.CreateEdit())
            {
                edit.Insert(pos, llmResponse);
                edit.Apply();

                var newCaretPosition = pos.Position + llmResponse.Length;
                docView.TextView.Caret.MoveTo(new SnapshotPoint(docView.TextBuffer.CurrentSnapshot, newCaretPosition));
            }
            numShifts = 0;
        }

        if (numCaps == capsThreshold)
        {
            var selection = docView.TextView.Selection;
            string queryText;

            if (!selection.IsEmpty)
            {
                queryText = selection.SelectedSpans[0].GetText();
            }
            else
            {
                queryText = currentLine.GetText();
            }

            string ftllmResponse = await QueryQwenFinetuned(queryText);
            ftllmResponse = "\n" + ftllmResponse.Replace("\\n", "\n");

            const string codeTag = "```";
            if (ftllmResponse.Contains(codeTag))
            {
                int firstTagIndex = ftllmResponse.IndexOf(codeTag);
                int lastTagIndex = ftllmResponse.LastIndexOf(codeTag);

                if (firstTagIndex != lastTagIndex)
                {
                    int firstNewlineAfterTag = ftllmResponse.IndexOf('\n', firstTagIndex);
                    if (firstNewlineAfterTag == -1)
                    {
                        firstNewlineAfterTag = ftllmResponse.Length;
                    }
                    int startIndex = firstNewlineAfterTag + 1;
                    int length = lastTagIndex - startIndex;
                    ftllmResponse = ftllmResponse.Substring(startIndex, length).Trim();
                }
            }

            using (var edit = docView.TextBuffer.CreateEdit())
            {
                if (!selection.IsEmpty)
                {
                    var selectedSpan = selection.SelectedSpans;
                    edit.Replace(selectedSpan[0], ftllmResponse);
                }
                else
                {
                    edit.Insert(pos, ftllmResponse);
                }
                edit.Apply();

                var newCaretPosition = pos.Position + ftllmResponse.Length;
                docView.TextView.Caret.MoveTo(new SnapshotPoint(docView.TextBuffer.CurrentSnapshot, newCaretPosition));
            }

            numCaps = 0;
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
    [ProvideAutoLoad(UIContextGuids80.NoSolution, PackageAutoLoadFlags.BackgroundLoad)]
    public sealed class NvAssistPackage : ToolkitPackage
    {
        protected override async Task InitializeAsync(CancellationToken cancellationToken, IProgress<ServiceProgressData> progress)
        {
            string assemblyLocation = Assembly.GetExecutingAssembly().Location;
            string installDirectory = Path.GetDirectoryName(assemblyLocation);

            string fineTuningDirectory = Path.Combine(installDirectory, "dist");
            string exePath = Path.Combine(fineTuningDirectory, "setupEnv.exe");

            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = exePath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            Process.Start(startInfo);
        }
    }
}