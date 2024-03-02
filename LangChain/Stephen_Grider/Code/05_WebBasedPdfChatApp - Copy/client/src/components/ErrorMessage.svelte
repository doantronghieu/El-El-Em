<script lang="ts">
	export let message = '';

	function setDoc(node: HTMLIFrameElement, message: string) {
		node.srcdoc = message;
		return {
			update(message: string) {
				node.srcdoc = message;
			}
		};
	}
</script>

<iframe class="h-full w-full" title="err" use:setDoc={message} />
