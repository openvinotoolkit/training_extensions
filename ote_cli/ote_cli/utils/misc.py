import logging
import subprocess


def convert_bash_command_for_log(cmd):
    if not cmd:
        return ''
    if isinstance(cmd, list):
        cmd = ' '.join(cmd)
    cmd = cmd.replace(';', '; ')
    cmd = cmd.split()

    if len(cmd) == 1:
        return cmd[0]

    shift_str = ' ' * 4
    split_str = ' \\\n' + shift_str
    semicolon_split_str = ' \\\n'
    max_line_len = 40
    max_chunk_len_to_keep_line = 30
    min_line_len_to_split_big_chunk = 10

    cur_line = cmd[0]
    cmdstr = ''
    for chunk in cmd[1:]:
        assert chunk
        if len(cur_line) > max_line_len:
            cmdstr += cur_line + split_str
            cur_line = ''
        if cur_line and chunk.startswith('--'):
            cmdstr += cur_line + split_str
            cur_line = ''
        if cur_line and chunk.startswith('|'):
            cmdstr += cur_line + split_str
            cur_line = ''
        if (cur_line
            and len(chunk) > max_chunk_len_to_keep_line
            and len(cur_line) >= min_line_len_to_split_big_chunk):
            cmdstr += cur_line + split_str
            cur_line = ''

        if cur_line:
            cur_line += ' '
        cur_line += chunk

        if cur_line.endswith(';'):
            cmdstr += cur_line + semicolon_split_str
            cur_line = ''

    cmdstr += cur_line

    while cmdstr.endswith(' ') or cmdstr.endswith('\n'):
        cmdstr = cmdstr[:-1]
    return cmdstr


def log_shell_cmd(cmd, prefix='Running through shell cmd'):
    cmdstr = convert_bash_command_for_log(cmd)
    logging.debug(f'{prefix}\n`{cmdstr}\n`')


def run_through_shell(cmd, verbose=True, check=True):
    assert isinstance(cmd, str)
    log_shell_cmd(cmd)
    std_streams_args = {} if verbose else {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
    return subprocess.run(cmd,
                          shell=True,
                          check=check,
                          executable="/bin/bash",
                          **std_streams_args)
