# Developer setup notes

## `.log/` symlink

`.log/` is excluded from version control (`.gitignore`) and backed up via a private folder
synced through Nextcloud (or any other private cloud). On a fresh clone, `.log/` will be
missing and must be recreated as a symlink pointing to that private folder.

**Setup on Windows (requires admin privileges or Developer Mode):**

1. Create (or locate) the target folder in your cloud sync directory, e.g.:
   `<Nextcloud-root>\insitupy\.log`
2. Open PowerShell **as Administrator** and run:
   ```powershell
   New-Item -ItemType SymbolicLink `
     -Path "<repo-root>\.log" `
     -Target "<cloud-sync-root>\insitupy\.log"
   ```
3. Verify: `Get-Item "<repo-root>\.log" | Select-Object LinkType, Target`

Replace `<repo-root>` with the path to this repository and `<cloud-sync-root>` with the root
of your personal cloud sync folder.
