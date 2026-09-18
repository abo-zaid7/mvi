# Untouched copy of the original Task 3 interface

These four files are the interface exactly as it was before the two extra designs
(Clinical workstation and Aurora) were added. They are here as a safety net.

| File | Goes back to |
|---|---|
| `index.html` | `../../templates/index.html` |
| `style.css` | `../../static/css/style.css` |
| `app.js` | `../../static/js/app.js` |
| `i18n.js` | `../../static/js/i18n.js` |

Nothing has to be restored to use the original design - choosing **Original** under
Display, or opening `http://127.0.0.1:5050/?skin=original`, already loads it, because
the skins are extra stylesheets layered on top of `style.css` rather than replacements
for it. `style.css` itself was never edited; `diff` it against the copy here and it
comes back empty.

To go back to the original design permanently, from inside `task3/`:

    cp backups/gui-original/index.html templates/index.html
    cp backups/gui-original/style.css  static/css/style.css
    cp backups/gui-original/app.js     static/js/app.js
    cp backups/gui-original/i18n.js    static/js/i18n.js
    rm static/css/skin-clinical.css static/css/skin-aurora.css

That removes the design chooser along with the two designs.
