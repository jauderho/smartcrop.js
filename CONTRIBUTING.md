# How to contribute

## Reporting Bugs

When reporting a bug please provide **all** the information needed to quickly reproduce the problem.

This includes:

- The browser & version used
- A URL to reproduce the error
- Other steps other necessary to reproduce the error

If I can't reproduce your problem it will likely not get fixed.

## Contributing Code

### Before Hacking

To avoid wasting your time please communicate before starting to hack
on a new feature / change.
Creating an issue describing your plans is a good start.

### Before submitting a pull request

- Try to make your code look like the code around it.
- Provide tests where appropriate.
- Run `bun fmt` and `bun run lint` on the files you changed, fix any issues related to your changes.
- Run the tests, make sure they are green

### Please Avoid

- Reformat my code to fit your personal preferences.

### License

By contributing your code, you agree to license your contribution under the MIT License.

## Setting up the development environment

Before you can start working you'll need to have [bun](https://bun.sh) and git installed.

```
$ git clone
$ cd smartcrop.js
$ bun install
$ bun run start
```

I work on GNU/Linux. Things might not work on windows.

## Running the tests

Run `bun test` for the unit tests and `bun check` for the type check.

To look at the crops visually, run `bun run start` and open
<http://localhost:8000/examples/testsuite.html>.

## Code of Conduct

None. Just be who you are.
