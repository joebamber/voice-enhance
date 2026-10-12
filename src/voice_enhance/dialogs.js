// Native Open/Save panels for voice-enhance --gui (run with osascript -l JavaScript).
//
// Started once with the GUI and kept running: launching osascript as a GUI app
// for each click took ~4 s before the panel appeared; a running helper shows it
// in ~0.1 s. Reads one JSON request per line on stdin, answers one per line.
//   {"save": false, "prompt": "...", "folder": "/dir"}            -> {"path": "/file" | null}
//   {"save": true, "prompt": "...", "folder": "/dir", "name": "x"} -> {"path": "/file" | null}

ObjC.import('AppKit');

const app = $.NSApplication.sharedApplication;
app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);  // no Dock icon
const stdin = $.NSFileHandle.fileHandleWithStandardInput;
const stdout = $.NSFileHandle.fileHandleWithStandardOutput;
const reply = obj => stdout.writeData($(JSON.stringify(obj) + '\n').dataUsingEncoding($.NSUTF8StringEncoding));

function ask(req) {
  const panel = req.save ? $.NSSavePanel.savePanel : $.NSOpenPanel.openPanel;
  panel.message = $(req.prompt || '');
  if (req.folder) panel.directoryURL = $.NSURL.fileURLWithPath(req.folder);
  if (req.save && req.name) panel.nameFieldStringValue = $(req.name);
  // come to the front over the browser, then hand focus back to it afterwards
  const previous = $.NSWorkspace.sharedWorkspace.frontmostApplication;
  app.activateIgnoringOtherApps(true);
  const ok = panel.runModal == $.NSModalResponseOK;
  const path = ok ? panel.URL.path.js : null;
  panel.orderOut(null);
  if (!previous.isNil()) previous.activateWithOptions($.NSApplicationActivateIgnoringOtherApps);
  settle();
  return path;
}

// Between requests the main thread sits blocked on stdin and handles no events,
// so a panel still on screen is a window that never answers: macOS shows it as
// a frozen beachball region. Let AppKit finish closing it before blocking again.
function settle() {
  const until = $.NSDate.dateWithTimeIntervalSinceNow(0.3);
  while (until.timeIntervalSinceNow > 0) {
    const ev = app.nextEventMatchingMaskUntilDateInModeDequeue(
      $.NSEventMaskAny, $.NSDate.dateWithTimeIntervalSinceNow(0.02), $.NSDefaultRunLoopMode, true);
    if (!ev.isNil()) app.sendEvent(ev);
  }
}

reply({ ready: true });
let pending = '';
for (;;) {
  const data = stdin.availableData;  // blocks until the server writes
  if (data.length == 0) break;  // server quit
  pending += $.NSString.alloc.initWithDataEncoding(data, $.NSUTF8StringEncoding).js;
  let nl;
  while ((nl = pending.indexOf('\n')) >= 0) {
    const line = pending.slice(0, nl);
    pending = pending.slice(nl + 1);
    try {
      reply({ path: ask(JSON.parse(line)) });
    } catch (e) {
      reply({ path: null, error: String(e) });
    }
  }
}
